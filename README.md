# ANN_Force biasing force plugin

This is the ANN_Force biasing force plugin for OpenMM (https://github.com/pandegroup/openmm) and accelerated sampling with data-augmented autoencoders framework (https://github.com/weiHelloWorld/accelerated_sampling_with_autoencoder).

## Dependency

OpenMM simulation pacakge: https://github.com/pandegroup/openmm

SWIG C/C++ wrapper: http://www.swig.org/

CMake: https://cmake.org/

## Installation

```bash
mkdir build
cd build
ccmake ..
```

Modify CMake installation settings as needed.  Turn on CUDA option if you would like to run simulation on CUDA.  Then run

```bash
make install
make PythonInstall
```

Root permission may be needed.

## Quick start

This package should be used together with OpenMM simulation package and training results of neural networks.  A good starting point would be tests in this framework: https://github.com/weiHelloWorld/accelerated_sampling_with_autoencoder.

## Testing



## TODO

- serialization?

- better synchronization for CUDA implementation?

## Contact

For any questions, feel free to contact weichen9@illinois.edu or open a github issue.

