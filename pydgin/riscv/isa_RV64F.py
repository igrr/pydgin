# =======================================================================
# isa_RV64F.py
# =======================================================================
'RISC-V instructions for the single-precision floating point extension.'

import softfloat as sfp
from pydgin.riscv.utils import signed

# =======================================================================
# Instruction Encodings
# =======================================================================

encodings = [

    ['fcvt_l_s', '110000000010xxxxxxxxxxxxx1010011'],
    ['fcvt_lu_s', '110000000011xxxxxxxxxxxxx1010011'],
    ['fcvt_s_l', '110100000010xxxxxxxxxxxxx1010011'],
    ['fcvt_s_lu', '110100000011xxxxxxxxxxxxx1010011'],

]


# =======================================================================
# Instruction Definitions
# =======================================================================

def execute_fcvt_l_s(s, inst):
    s.rf[inst.rd] = sfp.f32_to_i64(s.fp[inst.rs1], inst.rm, True)
    s.fcsr = sfp.get_flags()
    sfp.set_flags(0)
    s.pc += 4


def execute_fcvt_lu_s(s, inst):
    s.rf[inst.rd] = sfp.f32_to_ui64(s.fp[inst.rs1], inst.rm, True)
    s.fcsr = sfp.get_flags()
    sfp.set_flags(0)
    s.pc += 4


def execute_fcvt_s_l(s, inst):
    a = signed(s.rf[inst.rs1], 64)
    s.fp[inst.rd] = sfp.i64_to_f32(a)
    s.fcsr = sfp.get_flags()
    sfp.set_flags(0)
    s.pc += 4


def execute_fcvt_s_lu(s, inst):
    a = s.rf[inst.rs1]
    s.fp[inst.rd] = sfp.ui64_to_f32(a)
    s.fcsr = sfp.get_flags()
    sfp.set_flags(0)
    s.pc += 4
