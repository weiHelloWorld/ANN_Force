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

#include "CudaANN_Kernels.h"
#include "CudaANN_KernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"

using namespace OpenMM;
using namespace std;

class CudaANN_ForceInfo : public CudaForceInfo {   // TODO: what to do with this part?
public:
    CudaANN_ForceInfo(const ANN_Force& force) : force(force) {
    }
    int getNumParticleGroups() {
        // return force.getNumBonds();
        return 1;
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        // int particle1, particle2;
        // double length, k;
        // force.getBondParameters(index, particle1, particle2, length, k);
        // particles.resize(2);
        // particles[0] = particle1;
        // particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        // int particle1, particle2;
        // double length1, length2, k1, k2;
        // force.getBondParameters(group1, particle1, particle2, length1, k1);
        // force.getBondParameters(group2, particle1, particle2, length2, k2);
        // return (length1 == length2 && k1 == k2);
        return true;
    }
private:
    const ANN_Force& force;
};

CudaCalcANN_ForceKernel::~CudaCalcANN_ForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CudaCalcANN_ForceKernel::initialize(const System& system, const ANN_Force& force) {
    cu.setAsCurrent();
    // index_of_backbone_atoms = force.get_index_of_backbone_atoms();
    // layer_types = force.get_layer_types();
    // values_of_biased_nodes = force.get_values_of_biased_nodes();
    potential_center = force.get_potential_center();
    force_constant = force.get_force_constant();
    scaling_factor = force.get_scaling_factor();
    // data_type_in_input_layer = force.get_data_type_in_input_layer();
    // if (layer_types[NUM_OF_LAYERS - 2] != string("Circular")) {
    //     assert (potential_center.size() == num_of_nodes[NUM_OF_LAYERS - 1]);
    // }
    // else {
    //     assert (potential_center.size() * 2 == num_of_nodes[NUM_OF_LAYERS - 1]);
    // }
    
    // now deal with coefficients of connections
    // auto temp_coeff = force.get_coeffients_of_connections(); // FIXME: modify this initialization later
    // for (int ii = 0; ii < NUM_OF_LAYERS - 1; ii ++) {
    //     int num_of_rows, num_of_cols; // num of rows/cols for the coeff matrix of this connection
    //     num_of_rows = num_of_nodes[ii + 1];
    //     num_of_cols = num_of_nodes[ii];
    //     assert (num_of_rows * num_of_cols == temp_coeff[ii].size()); // check whether the size matches
    //     // create a 2d array to hold coefficients begin
    //     coeff[ii] = new double*[num_of_rows];
    //     for (int kk = 0; kk < num_of_rows; kk ++) {
    //         coeff[ii][kk] = new double[num_of_cols];
    //     }
    //     // creation end
    //     for (int jj = 0; jj < temp_coeff[ii].size(); jj ++) {
    //         coeff[ii][jj / num_of_cols][jj % num_of_cols] = temp_coeff[ii][jj];
    //     }
    // }
    // TODO: 
    // 1. store these parameters in device memory
    // 2. set up replacement text
    // 3. write .cu file
    // num_of_nodes = CudaArray::create<int>(cu, NUM_OF_LAYERS, "num_of_nodes_cuda");
    // num_of_nodes -> upload(force.get_num_of_nodes());

    numBonds = 1; 
    vector<vector<int> > atoms(numBonds, vector<int>(2));   // TODO: what is this?  2 is the number of atoms in this bond
    vector<float2> paramVector(numBonds);
    paramVector[0] = make_float2((float) potential_center[0], (float) force_constant);

    // convert to CUDA array
    num_of_nodes = convert_vector_to_CudaArray(force.get_num_of_nodes(), "1");
    params = convert_vector_to_CudaArray(paramVector, "2");
    index_of_backbone_atoms = convert_vector_to_CudaArray(force.get_index_of_backbone_atoms(), "3");
    {
        map<string, int> temp_replacement_layer_types;  // since string is not supported in CUDA kernel
        temp_replacement_layer_types["Linear"] = 0; temp_replacement_layer_types["Tanh"] = 1; temp_replacement_layer_types["Circular"] = 2; 
        vector<int> temp_layer_types(NUM_OF_LAYERS - 1);
        for (int ii = 0; ii < NUM_OF_LAYERS - 1; ii ++) {
            temp_layer_types[ii] = temp_replacement_layer_types[force.get_layer_types()[ii]];
        }
        layer_types = convert_vector_to_CudaArray(temp_layer_types, "4");
    }
    
    // replace text in .cu file
    map<string, string> replacements;  
    replacements["PARAMS"] = cu.getBondedUtilities().addArgument(params->getDevicePointer(), "float2");  
    replacements["PARAMS_1"] = cu.getBondedUtilities().addArgument(num_of_nodes->getDevicePointer(), "float");  

    cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CudaANN_KernelSources::ANN_Force, replacements), force.getForceGroup());
    cu.addForce(new CudaANN_ForceInfo(force));
}

double CudaCalcANN_ForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CudaCalcANN_ForceKernel::copyParametersToContext(ContextImpl& context, const ANN_Force& force) {
    // cu.setAsCurrent();
    // int numContexts = cu.getPlatformData().contexts.size();
    // int startIndex = cu.getContextIndex()*force.getNumBonds()/numContexts;
    // int endIndex = (cu.getContextIndex()+1)*force.getNumBonds()/numContexts;
    // if (numBonds != endIndex-startIndex)
    //     throw OpenMMException("updateParametersInContext: The number of bonds has changed");
    // if (numBonds == 0)
    //     return;
    
    // // Record the per-bond parameters.
    
    // vector<float2> paramVector(numBonds);
    // for (int i = 0; i < numBonds; i++) {
    //     int atom1, atom2;
    //     double length, k;
    //     force.getBondParameters(startIndex+i, atom1, atom2, length, k);
    //     paramVector[i] = make_float2((float) length, (float) k);
    // }
    // params->upload(paramVector);
    
    // // Mark that the current reordering may be invalid.
    
    // cu.invalidateMolecules();
}
