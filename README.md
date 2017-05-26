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

```bash
root_dir=openmmapi/tests
cd ${root_dir}
echo "running tests for numerical calculations..."
make test_ANN_package
./test_ANN_package
echo "running test for Python wrapper..."
python test_Python_wrapper.py
```

## Implementation details for CUDA platform

Here are some implementation details for CUDA platform if you are interested:

- Parallelization is realized by duplicating a single `ANN_Force` to get multiple copies for many threads.  By doing this, it can be well integrated into implementation of `CudaBondedUtilities` class (a force can only be assigned to one thread at most by default, that is why we need to get multiple copies for the force).

- We use **two-step code generation** to improve performance: 
    - use C++ to dynamically generate cuda framework code.  The purpose of this additional step is to
        - create code with minimal computation (remove unnecessary conditionals and loops), where C++ works for pre-preprocessing optimization
        - reduce memory access by replacing variable in shared memory with number literals
    - do preprocessing to get runnable cuda code.

- Inter-block synchronization is not done (I use all threads in one block to avoid this issue), but based on current benchmark, I do not expect it would accelerate computation too much.

- There is still room for optimization, in additional to synchronization issue, how to store variables in proper memory space and how to achieve workload balance among different threads would possibly make a difference.

## TODO

- serialization?

- better synchronization for CUDA implementation?

## Contact

For any questions, feel free to contact weichen9@illinois.edu or open a github issue.

