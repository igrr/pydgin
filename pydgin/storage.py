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


class RAM(MemoryLike):
    WORD_SIZE = 4

    def __init__(self, size_bytes: int):
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
        return [0, 0xff, 0xffff, 0, 0xffffffff][size_bytes] & (word >> shift)

    def write(self, addr: int, size_bytes: int, value: int) -> typing.Optional[MemError]:
        if size_bytes not in [1, 2, 4]:
            return MemError()
        if addr % size_bytes != 0:
            return MemError()
        word_addr = addr // self.WORD_SIZE
        old_word = self.words[word_addr]
        shift = (addr % self.WORD_SIZE) * 8
        mask = [0, 0xff, 0xffff, 0, 0xffffffff][size_bytes] << shift
        word = (old_word & (~mask)) | ((value << shift) & mask)
        self.words[word_addr] = word

    def size(self) -> int:
        return self.size_bytes


class MMIO(MemoryLike):
    WORD_SIZE = 4

    def __init__(self, size_bytes: int, read_handlers: typing.Dict = None, write_handlers: typing.Dict = None):
        assert size_bytes % self.WORD_SIZE == 0
        self.size_bytes = size_bytes
        self.size_words = size_bytes // self.WORD_SIZE
        self.read_handlers = read_handlers or {}
        self.write_handlers = write_handlers or {}

    def read(self, addr: int, size_bytes: int) -> typing.Union[int, MemError]:
        if size_bytes != self.WORD_SIZE:
            return MemError()
        if addr % self.WORD_SIZE != 0:
            return MemError()
        handler = self.read_handlers.get(addr, lambda x: 0)
        return handler(addr)

    def write(self, addr: int, size_bytes: int, value: int) -> typing.Optional[MemError]:
        if size_bytes != self.WORD_SIZE:
            return MemError()
        if addr % self.WORD_SIZE != 0:
            return MemError()
        handler = self.write_handlers.get(addr, lambda x: 0)
        handler(addr, value)

    def size(self) -> int:
        return self.size_bytes


class AddressSpace(MemoryLike):
    def __init__(self, regions: typing.List[typing.Tuple[int, MemoryLike]]):
        assert regions
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
        return self.regions[region_index][1].read(relative_addr, size_bytes)

    def write(self, addr: int, size_bytes: int, value: int) -> typing.Optional[MemError]:
        err, region_index, relative_addr = self._find_region(addr)
        if err:
            return err
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

# -------------------------------------------------------------------------
# _WordMemory
# -------------------------------------------------------------------------
# Memory that uses ints instead of chars
class _WordMemory(object):
    def __init__(self, data=None, size=2 ** 10, suppress_debug=False):
        self.data = data if data else [r_uint32(0)] * (size >> 2)
        self.size = r_uint(len(self.data) << 2)
        self.debug = Debug()
        self.suppress_debug = suppress_debug

        # TODO: pass data_section to memory for bounds checking
        self.data_section = 0x00000000

    def bounds_check(self, addr, x):
        # check if the accessed data is larger than the memory size
        if addr > self.size:
            print("WARNING: %s accessing larger address than memory size. "
                  "addr=%s size=%s") % (x, pad_hex(addr), pad_hex(self.size))
            raise Exception()
        if addr == 0:
            print("WARNING: accessing null pointer!")
            raise Exception()

        # Special write checks
        if x == 'WR' and addr < r_uint(self.data_section):
            print("WARNING: %s writing address below .data section!!!. "
                  "addr=%s size=%s") % (x, pad_hex(addr), pad_hex(self.data_section))
            raise Exception()

    def read(self, start_addr, num_bytes):
        assert 0 < num_bytes <= 4
        start_addr = r_uint(start_addr)
        word = start_addr >> 2
        byte = start_addr & 0b11

        if self.debug.enabled("mem") and not self.suppress_debug:
            print(':: RD.MEM[%s] = ' % pad_hex(start_addr))
        if self.debug.enabled("memcheck") and not self.suppress_debug:
            self.bounds_check(start_addr, 'RD')

        value = 0
        if num_bytes == 4:  # TODO: byte should only be 0 (only aligned)
            value = widen(self.data[word])
        elif num_bytes == 2:  # TODO: byte should only be 0, 1, 2, not 3
            mask = 0xFFFF << (byte * 8)
            value = (widen(self.data[word]) & mask) >> (byte * 8)
        elif num_bytes == 1:
            mask = 0xFF << (byte * 8)
            value = (widen(self.data[word]) & mask) >> (byte * 8)
        else:
            raise Exception('Invalid num_bytes: %d!' % num_bytes)

        if self.debug.enabled("mem"):
            print('%s' % pad_hex(value))

        return r_uint(value)

    # this is instruction read, which is otherwise identical to read. The
    # only difference is the elidable annotation, which we assume the
    # instructions are not modified (no side effects, assumes the addresses
    # correspond to the same instructions)
    def iread(self, start_addr, num_bytes):
        assert start_addr & 0b11 == 0  # only aligned accesses allowed
        return r_uint(widen(self.data[start_addr >> 2]))

    def write(self, start_addr, num_bytes, value):
        assert 0 < num_bytes <= 4
        start_addr = r_uint(start_addr)
        value = r_uint(value)
        word = start_addr >> 2
        byte = start_addr & 0b11

        if self.debug.enabled("memcheck") and not self.suppress_debug:
            self.bounds_check(start_addr, 'WR')

        if num_bytes == 4:  # TODO: byte should only be 0 (only aligned)
            pass  # no masking needed
        elif num_bytes == 2:  # TODO: byte should only be 0, 1, 2, not 3
            mask = ~(0xFFFF << (byte * 8)) & r_uint(0xFFFFFFFF)
            value = (widen(self.data[word]) & mask) \
                    | ((value & 0xFFFF) << (byte * 8))
        elif num_bytes == 1:
            mask = ~(0xFF << (byte * 8)) & r_uint(0xFFFFFFFF)
            value = (widen(self.data[word]) & mask) \
                    | ((value & 0xFF) << (byte * 8))
        else:
            raise Exception('Invalid num_bytes: %d!' % num_bytes)

        if self.debug.enabled("mem") and not self.suppress_debug:
            print(':: WR.MEM[%s] = %s' % (pad_hex(start_addr),
                                          pad_hex(value)))
        self.data[word] = r_uint32(value)


# -----------------------------------------------------------------------
# _ByteMemory
# -----------------------------------------------------------------------
class _ByteMemory(object):
    def __init__(self, data=None, size=2 ** 10, suppress_debug=False):
        self.data = data if data else [' '] * size
        self.size = len(self.data)
        self.debug = Debug()
        self.suppress_debug = suppress_debug

    def bounds_check(self, addr):
        # check if the accessed data is larger than the memory size
        if addr > self.size:
            print("WARNING: accessing larger address than memory size. " + \
                  "addr=%s size=%s" % (pad_hex(addr), pad_hex(self.size)))
        if addr == 0:
            print("WARNING: writing null pointer!")
            raise Exception()

    def read(self, start_addr, num_bytes):
        if self.debug.enabled("memcheck") and not self.suppress_debug:
            self.bounds_check(start_addr)
        value = 0
        if self.debug.enabled("mem") and not self.suppress_debug:
            print(':: RD.MEM[%s] = ' % pad_hex(start_addr))
        for i in range(num_bytes - 1, -1, -1):
            value = value << 8
            value = value | ord(self.data[start_addr + i])
        if self.debug.enabled("mem") and not self.suppress_debug:
            print('%s' % pad_hex(value))
        return value

    # this is instruction read, which is otherwise identical to read. The
    # only difference is the elidable annotation, which we assume the
    # instructions are not modified (no side effects, assumes the addresses
    # correspond to the same instructions)
    def iread(self, start_addr, num_bytes):
        value = 0
        for i in range(num_bytes - 1, -1, -1):
            value = value << 8
            value = value | ord(self.data[start_addr + i])
        return value

    def write(self, start_addr, num_bytes, value):
        if self.debug.enabled("memcheck") and not self.suppress_debug:
            self.bounds_check(start_addr)
        if self.debug.enabled("mem") and not self.suppress_debug:
            print(':: WR.MEM[%s] = %s' % (pad_hex(start_addr),
                                          pad_hex(value)))
        for i in range(num_bytes):
            self.data[start_addr + i] = chr(value & 0xFF)
            value = value >> 8


# -----------------------------------------------------------------------
# _SparseMemory
# -----------------------------------------------------------------------

class _SparseMemory(object):
    _immutable_fields_ = ["BlockMemory", "block_size", "addr_mask",
                          "block_mask"]

    def __init__(self, BlockMemory, block_size=2 ** 10):
        self.BlockMemory = BlockMemory
        self.block_size = block_size
        self.addr_mask = block_size - 1
        self.block_mask = 0xffffffff ^ self.addr_mask
        self.debug = Debug()
        print("sparse memory size %x addr mask %x block mask %x" \
              % (self.block_size, self.addr_mask, self.block_mask))
        # blocks     = []
        self.block_dict = {}
        self.debug = Debug()

    def add_block(self, block_addr):
        # print "adding block: %x" % block_addr
        self.block_dict[block_addr] = self.BlockMemory(size=self.block_size,
                                                       suppress_debug=True)

    def get_block_mem(self, block_addr):
        # block_idx  = block_dict[
        if block_addr not in self.block_dict:
            self.add_block(block_addr)
        block_mem = self.block_dict[block_addr]
        return block_mem

    def iread(self, start_addr, num_bytes):
        end_addr = start_addr + num_bytes - 1

        block_addr = self.block_mask & start_addr
        block_mem = self.get_block_mem(block_addr)
        # For mixed-width ISAs, the start_addr is not necessarily
        # word-aligned, and can cross block memory boundaries. If there is
        # such a case, we have two instruction reads and then form the word
        # for it
        block_end_addr = self.block_mask & end_addr
        if block_addr == block_end_addr:
            return block_mem.iread(start_addr & self.addr_mask, num_bytes)
        else:
            num_bytes1 = min(self.block_size - (start_addr & self.addr_mask),
                             num_bytes)
            num_bytes2 = num_bytes - num_bytes1

            block_mem1 = block_mem
            block_mem2 = self.get_block_mem(block_end_addr)
            value1 = block_mem1.iread(start_addr & self.addr_mask, num_bytes1)
            value2 = block_mem2.iread(0, num_bytes2)
            value = value1 | (value2 << (num_bytes1 * 8))
            # print "nb1", num_bytes1, "nb2", num_bytes2, \
            #      "ba1", hex(block_addr), "ba2", hex(block_end_addr), \
            #      "v1", hex(value1), "v2", hex(value2), "v", hex(value)
            return value

    def read(self, start_addr, num_bytes):
        if self.debug.enabled("mem"):
            print(':: RD.MEM[%s] = ' % pad_hex(start_addr))
        block_addr = self.block_mask & start_addr
        block_mem = self.get_block_mem(block_addr)
        value = block_mem.read(start_addr & self.addr_mask, num_bytes)
        if self.debug.enabled("mem"):
            print('%s' % pad_hex(value))
        return value

    def write(self, start_addr, num_bytes, value):
        if self.debug.enabled("mem"):
            print(':: WR.MEM[%s] = %s' % (pad_hex(start_addr),
                                          pad_hex(value)))
        block_addr = self.block_mask & start_addr
        block_mem = self.get_block_mem(block_addr)
        block_mem.write(start_addr & self.addr_mask, num_bytes, value)
