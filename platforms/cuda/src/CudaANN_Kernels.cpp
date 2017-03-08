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
    // int numContexts = cu.getPlatformData().contexts.size();
    // int startIndex = cu.getContextIndex()*force.getNumBonds()/numContexts;
    // int endIndex = (cu.getContextIndex()+1)*force.getNumBonds()/numContexts;
    // numBonds = endIndex-startIndex;
    // if (numBonds == 0)
    //     return;
    numBonds = 1; 
    vector<vector<int> > atoms(numBonds, vector<int>(2));
    params = CudaArray::create<float2>(cu, numBonds, "bondParams");
    vector<float2> paramVector(numBonds);
    for (int i = 0; i < numBonds; i++) {
        double force_constant = force.get_force_constant();
        auto potential_center = force.get_potential_center()[0];   // TODO: temp, just get first component for test
        cout << potential_center;
        paramVector[i] = make_float2((float) potential_center, (float) force_constant);
    }
    params->upload(paramVector);   // Copy the values in a vector to the device memory.
    map<string, string> replacements;
    replacements["PARAMS"] = cu.getBondedUtilities().addArgument(params->getDevicePointer(), "float2");
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
