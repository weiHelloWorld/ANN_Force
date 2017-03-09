#ifndef CUDA_ANN__KERNELS_H_
#define CUDA_ANN__KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "OpenMM_ANN.h"
#include "openmm/ANN_Kernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

namespace OpenMM {

/**
 * This kernel is invoked by ANN_Force to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcANN_ForceKernel : public CalcANN_ForceKernel {
public:
    CudaCalcANN_ForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu, const OpenMM::System& system) :
            CalcANN_ForceKernel(name, platform), hasInitializedKernel(false), cu(cu), system(system), params(NULL) {
    }
    ~CudaCalcANN_ForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the ANN_Force this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const ANN_Force& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the ANN_Force to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const ANN_Force& force);

    template <class T>
    OpenMM::CudaArray* convert_vector_to_CudaArray(vector<T> temp_vec, string name) {
        auto temp_cuda_array = CudaArray::create<T>(cu, temp_vec.size(), name);
        temp_cuda_array -> upload(temp_vec);   // Copy the values in a vector to the device memory.
        return temp_cuda_array;
    }

private:
    int numBonds;
    bool hasInitializedKernel;
    OpenMM::CudaContext& cu;
    const OpenMM::System& system;
    OpenMM::CudaArray* params;
    OpenMM::CudaArray* num_of_nodes;  // vector<int>
    OpenMM::CudaArray* index_of_backbone_atoms; // vector<int>
    OpenMM::CudaArray* layer_types; // vector<string>, should be converted to int array in CUDA kernel
    OpenMM::CudaArray* potential_center; // vector<double>
    OpenMM::CudaArray* force_constant;  // double, converted to vector<double> in kernel
    OpenMM::CudaArray* scaling_factor;  // double, converted to vector<double> in kernel
    double potential_energy;
    
    vector<double** > coeff = vector<double** >(NUM_OF_LAYERS - 1);  // each coeff of connection is a matrix
    vector<vector<double> > output_of_each_layer = vector<vector<double> >(NUM_OF_LAYERS);
    vector<vector<double> > input_of_each_layer = vector<vector<double> >(NUM_OF_LAYERS);
    vector<vector<double> > values_of_biased_nodes = vector<vector<double> >(NUM_OF_LAYERS - 1);
    int data_type_in_input_layer;
};

} // namespace OpenMM

#endif /*CUDA_ANN__KERNELS_H_*/
