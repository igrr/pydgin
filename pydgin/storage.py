# =======================================================================
# storage.py
# =======================================================================

from .debug import Debug, pad, pad_hex
from .utils import r_uint
from abc import ABC, abstractmethod
from array import array
import typing
import bisect

r_uint32 = int


def widen(value):
    return value


# -----------------------------------------------------------------------
# RegisterFile
# -----------------------------------------------------------------------
class RegisterFile(object):
    def __init__(self, constant_zero=True, num_regs=32, nbits=32):
        self.num_regs = num_regs
        self.regs = [r_uint(0)] * self.num_regs
        self.debug = Debug()
        self.nbits = nbits
        self.debug_nchars = nbits / 4

        if constant_zero:
            self._setitemimpl = self._set_item_const_zero
        else:
            self._setitemimpl = self._set_item

    def __getitem__(self, idx):
        if self.debug.enabled("rf"):
            print(':: RD.RF[%s] = %s' % (
                pad("%d" % idx, 2),
                pad_hex(self.regs[idx],
                        len=self.debug_nchars)))
        return self.regs[idx]

    def __setitem__(self, idx, value):
        value = r_uint(value)
        self._setitemimpl(idx, value)

    def _set_item(self, idx, value):
        self.regs[idx] = value
        if self.debug.enabled("rf"):
            print(':: WR.RF[%s] = %s' % (
                pad("%d" % idx, 2),
                pad_hex(self.regs[idx],
                        len=self.debug_nchars)))

    def _set_item_const_zero(self, idx, value):
        if idx != 0:
            self.regs[idx] = value
            if self.debug.enabled("rf"):
                print(':: WR.RF[%s] = %s' % (
                    pad("%d" % idx, 2),
                    pad_hex(self.regs[idx],
                            len=self.debug_nchars)))

    # -----------------------------------------------------------------------
    # print_regs
    # -----------------------------------------------------------------------
    # prints all registers (register dump)
    # per_row specifies the number of registers to display per row
    def print_regs(self, per_row=6):
        for c in range(0, self.num_regs, per_row):
            str = ""
            for r in range(c, min(self.num_regs, c + per_row)):
                str += "%s:%s " % (pad("%d" % r, 2),
                                   pad_hex(self.regs[r], len=(self.nbits / 4)))
            print(str)


# -----------------------------------------------------------------------
# Memory
# -----------------------------------------------------------------------
class MemError(object):
    pass


class MemoryLike(ABC):
    @abstractmethod
    def read(self, addr: int, size_bytes: int) -> typing.Union[int, MemError]:
        return 0

    @abstractmethod
    def write(self, addr: int, size_bytes: int, value: int) -> typing.Optional[MemError]:
        pass

    @abstractmethod
    def size(self) -> int:
        pass


class MemAccessRecord(object):
    def __init__(self, is_write: bool, addr: int, size_bytes: int, val: int):
        self.is_write = is_write
        self.addr = addr
        self.size_bytes = size_bytes
        self.val = val

    def __str__(self):
        return "{}{}@0x{:08x}=0x{:08x}".format("W" if self.is_write else "R",
                                               self.size_bytes,
                                               self.addr,
                                               self.val)


class Monitor(object):
    class NoMonitorCM(object):
        def __init__(self, m):
            self.m = m
            self.saved_access_list = None

        def __enter__(self):
            self.saved_access_list = self.m.access_list
            self.m.access_list = None

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.m.access_list = self.saved_access_list

    def __init__(self):
        self.access_list = None  # type: typing.Optional[typing.List[MemAccessRecord]]

    def monitor(self, is_write: bool, addr: int, size_bytes: int, val: int):
        if self.access_list is not None:
            self.access_list.append(MemAccessRecord(is_write, addr, size_bytes, val))

    def no_monitor(self) -> NoMonitorCM:
        return Monitor.NoMonitorCM(self)


class RAM(MemoryLike, Monitor):
    WORD_SIZE = 4

    def __init__(self, size_bytes: int):
        super(RAM, self).__init__()
        assert size_bytes % self.WORD_SIZE == 0
        self.size_bytes = size_bytes
        self.size_words = size_bytes // self.WORD_SIZE
        self.words = array('L', [0] * self.size_words)

    def read(self, addr: int, size_bytes: int) -> typing.Union[int, MemError]:
        if size_bytes not in [1, 2, 4]:
            return MemError()
        if addr % size_bytes != 0:
            return MemError()
        word_addr = addr // self.WORD_SIZE
        word = self.words[word_addr]
        shift = (addr % self.WORD_SIZE) * 8
        res = [0, 0xff, 0xffff, 0, 0xffffffff][size_bytes] & (word >> shift)
        self.monitor(False, addr, size_bytes, res)
        return res

    def write(self, addr: int, size_bytes: int, value: int) -> typing.Optional[MemError]:
        if size_bytes not in [1, 2, 4]:
            return MemError()
        if addr % size_bytes != 0:
            return MemError()
        self.monitor(True, addr, size_bytes, value)
        word_addr = addr // self.WORD_SIZE
        old_word = self.words[word_addr]
        shift = (addr % self.WORD_SIZE) * 8
        mask = [0, 0xff, 0xffff, 0, 0xffffffff][size_bytes] << shift
        word = (old_word & (~mask)) | ((value << shift) & mask)
        self.words[word_addr] = word

    def size(self) -> int:
        return self.size_bytes


class MMIO(MemoryLike, Monitor):
    WORD_SIZE = 4

    def __init__(self, size_bytes: int, read_handlers: typing.Dict = None, write_handlers: typing.Dict = None,
                 default_read_handler=None, default_write_handler=None,
                 valid_access_size: typing.List[int] = None,
                 valid_alignment: typing.List[int] = None):
        assert size_bytes % self.WORD_SIZE == 0
        super(MMIO, self).__init__()
        self.size_bytes = size_bytes
        self.size_words = size_bytes // self.WORD_SIZE
        self.read_handlers = read_handlers or {}
        self.write_handlers = write_handlers or {}
        def read_no_op(addr):
            pass
        def write_no_op(addr, val):
            pass

        self.default_read_handler = default_read_handler or read_no_op
        self.default_write_handler = default_write_handler or write_no_op
        self.valid_access_size = valid_access_size or [4]
        self.valid_alignment = valid_alignment or [0]

    def read(self, addr: int, size_bytes: int) -> typing.Union[int, MemError]:
        err = self._check_addr_size(addr, size_bytes)
        if err is not None:
            return err

        handler = self.read_handlers.get(addr, self.default_read_handler)
        res = handler(addr)
        self.monitor(False, addr, size_bytes, res)
        return res

    def write(self, addr: int, size_bytes: int, value: int) -> typing.Optional[MemError]:
        err = self._check_addr_size(addr, size_bytes)
        if err is not None:
            return err

        self.monitor(True, addr, size_bytes, value)
        handler = self.write_handlers.get(addr, self.default_write_handler)
        handler(addr, value)
        return None

    def _check_addr_size(self, addr: int, size_bytes: int) -> typing.Optional[MemError]:
        if size_bytes not in self.valid_access_size:
            return MemError()
        if addr % size_bytes not in self.valid_alignment:
            return MemError()
        return None

    def size(self) -> int:
        return self.size_bytes


class AddressSpace(MemoryLike, Monitor):
    def __init__(self, regions: typing.List[typing.Tuple[int, MemoryLike]]):
        assert regions
        super(AddressSpace, self).__init__()
        self.regions = sorted(regions, key=lambda p: p[0])
        self.regions_start = [p[0] for p in self.regions]
        self.regions_end = [p[0] + p[1].size() for p in self.regions]
        self.regions_count = len(self.regions)
        for i in range(self.regions_count - 1):
            assert self.regions_end[i] <= self.regions_start[i + 1]

        self.size_bytes = self.regions_end[len(self.regions_end) - 1]

    def size(self) -> int:
        return self.size_bytes

    def _find_region(self, addr: int) -> typing.Tuple[typing.Optional[MemError], int, int]:
        region_index = bisect.bisect_right(self.regions_start, addr)
        if region_index == 0:
            return MemError(), 0, 0
        region_index -= 1
        assert addr >= self.regions_start[region_index]
        if addr > self.regions_end[region_index]:
            return MemError(), 0, 0
        relative_addr = addr - self.regions_start[region_index]
        return None, region_index, relative_addr

    def read(self, addr: int, size_bytes: int) -> typing.Union[int, MemError]:
        err, region_index, relative_addr = self._find_region(addr)
        if err:
            return err
        res = self.regions[region_index][1].read(relative_addr, size_bytes)
        self.monitor(False, addr, size_bytes, res)
        return res

    def write(self, addr: int, size_bytes: int, value: int) -> typing.Optional[MemError]:
        err, region_index, relative_addr = self._find_region(addr)
        if err:
            return err
        self.monitor(True, addr, size_bytes, value)
        return self.regions[region_index][1].write(relative_addr, size_bytes, value)


class MemErrorCatcher(MemoryLike):
    def __init__(self, inner: MemoryLike):
        self.inner = inner

    def read(self, addr: int, size_bytes: int) -> typing.Union[int, MemError]:
        res = self.inner.read(addr, size_bytes)
        if isinstance(res, MemError):
            raise RuntimeError(str(res))
        return res

    def write(self, addr: int, size_bytes: int, value: int) -> typing.Optional[MemError]:
        res = self.inner.write(addr, size_bytes, value)
        if isinstance(res, MemError):
            raise RuntimeError(str(res))
        return None

    def size(self) -> int:
        return self.inner.size()
