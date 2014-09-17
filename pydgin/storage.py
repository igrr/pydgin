#=======================================================================
# storage.py
#=======================================================================

from rpython.rlib.jit import elidable, unroll_safe
from debug            import Debug, pad, pad_hex

#-----------------------------------------------------------------------
# RegisterFile
#-----------------------------------------------------------------------
class RegisterFile( object ):
  def __init__( self ):
    self.regs  = [0] * 32
    self.debug = Debug()
  def __getitem__( self, idx ):
    if self.debug.enabled( "rf" ):
      print ':: RD.RF[%s] = %s' % (
                          pad( "%d" % idx, 2 ),
                          pad_hex( self.regs[idx]) ),
    return self.regs[idx]
  def __setitem__( self, idx, value ):
    if idx != 0:
      self.regs[idx] = value
      if self.debug.enabled( "rf" ):
        print ':: WR.RF[%s] = %s' % (
                          pad( "%d" % idx, 2 ),
                          pad_hex( self.regs[idx] ) ),

  #-----------------------------------------------------------------------
  # print_regs
  #-----------------------------------------------------------------------
  # prints all registers (register dump)
  def print_regs( self ):
    num_regs = 32
    per_row  = 6
    for c in xrange( 0, num_regs, per_row ):
      str = ""
      for r in xrange( c, min( num_regs, c+per_row ) ):
        str += "%s:%s " % ( pad( "%d" % r, 2 ),
                            pad_hex( self.regs[r] ) )
      print str

#-----------------------------------------------------------------------
# Memory
#-----------------------------------------------------------------------
def Memory( data=None, size=2**10, byte_storage=False ):
  if byte_storage:
    return _ByteMemory( data, size )
  else:
    return _WordMemory( data, size )

#-------------------------------------------------------------------------
# _WordMemory
#-------------------------------------------------------------------------
# Memory that uses ints instead of chars
class _WordMemory( object ):
  def __init__( self, data=None, size=2**10 ):
    if not data:
      self.data = [0] * (size >> 2)
    else:
      self.data = data
    self.size = len( self.data ) >> 2

  @unroll_safe
  def read( self, start_addr, num_bytes ):
    assert 0 < num_bytes <= 4
    word = start_addr >> 2
    byte = start_addr &  0b11

    if   num_bytes == 4:  # TODO: byte should only be 0 (only aligned)
      return self.data[ word ]
    elif num_bytes == 2:  # TODO: byte should only be 0, 1, 2, not 3
      mask = 0xFFFF << (byte * 8)
      return (self.data[ word ] & mask) >> (byte * 8)
    elif num_bytes == 1:
      mask = 0xFF   << (byte * 8)
      return (self.data[ word ] & mask) >> (byte * 8)

    raise Exception('Not handled value for num_bytes')

  # this is instruction read, which is otherwise identical to read. The
  # only difference is the elidable annotation, which we assume the
  # instructions are not modified (no side effects, assumes the addresses
  # correspond to the same instructions)
  @elidable
  @unroll_safe
  def iread( self, start_addr, num_bytes ):
    assert start_addr & 0b11 == 0  # only aligned accesses allowed
    return self.data[ start_addr >> 2 ]

  @unroll_safe
  def write( self, start_addr, num_bytes, value ):
    assert 0 < num_bytes <= 4
    word = start_addr >> 2
    byte = start_addr &  0b11

    if   num_bytes == 4:  # TODO: byte should only be 0 (only aligned)
      self.data[ word ] = value
    elif num_bytes == 2:  # TODO: byte should only be 0, 1, 2, not 3
      mask = ~(0xFFFF << (byte * 8)) & 0xFFFFFFFF
      self.data[ word ] = ( self.data[ word ] & mask ) | \
                          ( (value & 0xFFFF) << (byte * 8) )
    elif num_bytes == 1:
      mask = ~(0xFF   << (byte * 8)) & 0xFFFFFFFF
      self.data[ word ] = ( self.data[ word ] & mask ) | \
                          ( (value & 0xFF  ) << (byte * 8) )
    else:
      raise Exception('Not handled value for num_bytes')

#-----------------------------------------------------------------------
# _ByteMemory
#-----------------------------------------------------------------------
class _ByteMemory( object ):
  def __init__( self, data=None, size=2**10 ):
    if not data:
      self.data = [' '] * size
    else:
      self.data = data
    self.size = len( self.data )
    self.debug = Debug()

  def bounds_check( self, addr ):
    # check if the accessed data is larger than the memory size
    if addr > self.size:
      print "WARNING: accessing larger address than memory size. " + \
            "addr=%s size=%s" % ( pad_hex( addr ), pad_hex( self.size ) )

  @unroll_safe
  def read( self, start_addr, num_bytes ):
    if self.debug.enabled( "memcheck" ):
      self.bounds_check( start_addr )
    value = 0
    if self.debug.enabled( "mem" ):
      print ':: RD.MEM[%s] = ' % pad_hex( start_addr ),
    for i in range( num_bytes-1, -1, -1 ):
      value = value << 8
      value = value | ord( self.data[ start_addr + i ] )
    if self.debug.enabled( "mem" ):
      print '%s' % pad_hex( value ),
    return value

  # this is instruction read, which is otherwise identical to read. The
  # only difference is the elidable annotation, which we assume the
  # instructions are not modified (no side effects, assumes the addresses
  # correspond to the same instructions)
  @elidable
  def iread( self, start_addr, num_bytes ):
    value = 0
    for i in range( num_bytes-1, -1, -1 ):
      value = value << 8
      value = value | ord( self.data[ start_addr + i ] )
    return value

  @unroll_safe
  def write( self, start_addr, num_bytes, value ):
    if self.debug.enabled( "memcheck" ):
      self.bounds_check( start_addr )
    if self.debug.enabled( "mem" ):
      print ':: WR.MEM[%s] = %s' % ( pad_hex( start_addr ),
                                     pad_hex( value ) ),
    for i in range( num_bytes ):
      self.data[ start_addr + i ] = chr(value & 0xFF)
      value = value >> 8

