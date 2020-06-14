This repository contains a stripped-down fork of [Pydgin](https://github.com/cornell-brg/pydgin).

The original codebase has a large number of features, but hasn't been maintained for a while, and contains a number of hacks.
Because of these hacks it is hard to use Pydgin as a library.
My use case was to produce "ground truth" data about expected CPU behavior inside a SoC testbench (based on nMigen).
Therefore, only the following functionality is preserved:

* RISC-V and ARM architectures.
* Instruction decoding, instruction execution, register files.

The following features have been removed:

- syscalls and bootstrapping
- Top-level simulation driver
- JIT
- Custom ELF parser
- softfloat (hence RISC-V `F` and `D` extensions do not work)

Other changes include general cleanup and Python 3 compatibility.
