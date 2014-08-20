#=======================================================================
# syscalls.py
#=======================================================================
# Implementations of emulated syscalls. Call numbers were borrowed from
# the following files:
#
# - https://github.com/cornell-brg/maven-sim-isa/blob/master/common/syscfg.h
# - https://github.com/cornell-brg/pyparc/blob/master/pkernel/pkernel/syscfg.h
# - https://github.com/cornell-brg/maven-sys-xcc/blob/master/src/libgloss/maven/machine/syscfg.h
# - https://github.com/cornell-brg/gem5-mcpat/blob/master/src/arch/mips/linux/process.cc
#
# Implementation details from the Maven proxy kernel:
#
# - https://github.com/cornell-brg/pyparc/blob/master/pkernel/pkernel/pkernel.c#L463-L544
# - https://github.com/cornell-brg/pyparc/blob/master/pkernel/pkernel/pkernel-xcpthandler.S#L78-L89
#
#   // Jump to C function to handle syscalls
#
#   move    $a3, $a2 // arg2
#   move    $a2, $a1 // arg1
#   move    $a1, $a0 // arg0
#   move    $a0, $v0 // syscall number
#   la      $t0, handle_syscall
#   jalr    $t0
#
#   // Restore user context
#
#   move    $k0, $sp
#
# Details for newlib syscalls from the Maven cross compiler:
#
# - https://github.com/cornell-brg/maven-sys-xcc/blob/master/src/libgloss/maven/syscalls.c
#
# According to Berkin, only the following syscalls are needed by pbbs:
#
# - 2 read
# - 3 write
# - 4 open
# - 5 close
# - 8 lseek

from isa import reg_map

v0 = reg_map['a0']
a0 = reg_map['a0']
a1 = reg_map['a1']
a2 = reg_map['a2']

#-----------------------------------------------------------------------
# TODO
#-----------------------------------------------------------------------
def syscall_read( s ):
  raise Exception('read unimplemented!')
def syscall_write( s ):
  raise Exception('write unimplemented!')
def syscall_open( s ):
  raise Exception('open unimplemented!')
def syscall_close( s ):
  raise Exception('close unimplemented!')
def syscall_lseek( s ):
  raise Exception('close unimplemented!')

#-----------------------------------------------------------------------
# brk
#-----------------------------------------------------------------------
# http://stackoverflow.com/questions/6988487/what-does-brk-system-call-do
def syscall_brk( s ):
  kernel_addr = s.rf[ a0 ]
  user_addr   = s.rf[ a1 ]

  # first call to brk initializes the breakk_point address (end of heap)
  # TODO: initialize in pisa-sim::syscall_init()!
  if s.break_point == 0:
    s.break_point = user_addr

  # if kernel_addr is not null, set a new break_point
  if kernel_addr != 0:
    s.break_point = kernel_addr

  # return the break_point value
  s.rf[ v0 ] = s.break_point

#-----------------------------------------------------------------------
# numcores
#-----------------------------------------------------------------------
def syscall_numcores( s ):
  # always return 1 until multicore is implemented!
  s.rf[ v0 ] = 1

#-----------------------------------------------------------------------
# syscall number mapping
#-----------------------------------------------------------------------
syscall_funcs = {
#   0: syscall,       # unimplemented_func
#   1: exit,
    2: syscall_read,
    3: syscall_write,
    4: syscall_open,
    5: syscall_close,
#   6: link,
#   7: unlink,
    8: syscall_lseek,
#   9: fstat,
#  10: stat,
   11: syscall_brk,
 4000: syscall_numcores,

#4001: sendam,
#4002: bthread_once,
#4003: bthread_create,
#4004: bthread_delete,
#4005: bthread_setspecific,
#4006: bthread_getspecific,
#4007: yield,
}
