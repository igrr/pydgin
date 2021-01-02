# =======================================================================
# misc.py
# =======================================================================

import re

from elftools.elf import elffile
# These are used in the string exec'ed below:
from .utils import *


# -----------------------------------------------------------------------
# FatalError
# -----------------------------------------------------------------------
# We use our own exception class to terminate execution on a fatal error.

class FatalError(Exception):

    def __init__(self, msg):
        self.msg = msg


# -----------------------------------------------------------------------
# NotImplementedInstError
# -----------------------------------------------------------------------
# Error for not implemented instructions

class NotImplementedInstError(FatalError):

    def __init__(self, msg="Instruction not implemented"):
        FatalError.__init__(self, msg)


# -----------------------------------------------------------------------
# load_program
# -----------------------------------------------------------------------
def load_program(fp, mem, alignment=0, is_64bit=False):
    ef = elffile.ELFFile(fp)
    entrypoint = ef.header['e_entry']

    for segment in ef.iter_segments():
        if segment['p_type'] == "PT_LOAD":
            for section in ef.iter_sections():
                if not segment.section_in_segment(section) or section['sh_type'] == 'SHT_NOBITS':
                    continue
                data = section.data()
                print("Loading {} bytes from {}".format(len(data), section.name))
                load_addr = section['sh_addr']
                for i, b in enumerate(data):
                    mem.write(load_addr + i, 1, b)

    return entrypoint


# -----------------------------------------------------------------------
# create_risc_decoder
# -----------------------------------------------------------------------
def create_risc_decoder(encodings, isa_globals, debug=False):
    # removes all characters other than '0', '1', and 'x'
    def remove_ignored_chars(enc):
        return [enc[0], re.sub('[^01x]', '', enc[1])]

    encodings = list(map(remove_ignored_chars, encodings))

    def split_encodings(enc):
        return [x for x in re.split('(x+)', enc) if x]

    bit_fields = [split_encodings(x[1]) for x in encodings]

    decoder = ''
    for i, inst in enumerate(bit_fields):
        # print i, encodings[i][0], inst
        bit = 0
        conditions = []
        for field in reversed(inst):
            nbits = len(field)
            if field[0] != 'x':
                mask = (1 << nbits) - 1
                cond = '(inst >> {}) & r_uint(0x{:X}) == r_uint(0b{})'.format(bit, mask, field)
                conditions.append(cond)
            bit += nbits
        decoder += 'if   ' if i == 0 else '  elif '
        decoder += ' and '.join(reversed(conditions)) + ':\n'
        if debug:
            decoder += '    return "{0}", execute_{0}\n'.format(encodings[i][0])
        else:
            decoder += '    return execute_{}\n'.format(encodings[i][0])

    source = '''
# @elidable
def decode( inst ):
  {decoder_tree}
  else:
    raise FatalError('Invalid instruction 0x%x!' % inst )
  '''.format(decoder_tree=decoder)

    # print(source)
    environment = dict(globals().items())
    environment.update(isa_globals.items())
    exec(source, environment)

    return environment['decode']
