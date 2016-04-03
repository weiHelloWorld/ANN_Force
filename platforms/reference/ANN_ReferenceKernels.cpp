#include "ANN_ReferenceKernels.h"
#include "openmm/internal/ANN_ForceImpl.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "../../openmmapi/src/ANN_Force.cpp"

#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->velocities);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

static RealVec& extractBoxSize(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *(RealVec*) data->periodicBoxSize;
}

static RealVec* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (RealVec*) data->periodicBoxVectors;
}

// ***************************************************************************

ReferenceCalcANN_ForceKernel::ReferenceCalcANN_ForceKernel(std::string name, const Platform& platform, const System& system) : 
                CalcANN_ForceKernel(name, platform), system(system) {
}

ReferenceCalcANN_ForceKernel::~ReferenceCalcANN_ForceKernel() {
}

void ReferenceCalcANN_ForceKernel::initialize(const System& system, const ANN_Force& force) {
    num_of_nodes = force.get_num_of_nodes();
    index_of_backbone_atoms = force.get_index_of_backbone_atoms();
    coeff = force.get_coeffients_of_connections();
    layer_types = force.get_layer_types();
    return;
}

double ReferenceCalcANN_ForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& posData   = extractPositions(context);
    vector<RealVec>& forceData = extractForces(context);
    RealOpenMM energy      = calculateForceAndEnergy(posData, forceData); 
                                // the output of force for each atom is stored in forceData
    // TODO: implement calculation of force (of each atom) and energy here
    return static_cast<double>(energy);
}

void ReferenceCalcANN_ForceKernel::copyParametersToContext(ContextImpl& context, const ANN_Force& force) {
    // if (numBonds != force.getNumBonds())
    //     throw OpenMMException("updateParametersInContext: The number of bonds has changed");

    // // Record the values.

    // for (int i = 0; i < numBonds; ++i) {
    //     int particle1Index, particle2Index;
    //     double lengthValue, kValue;
    //     force.getBondParameters(i, particle1Index, particle2Index, lengthValue, kValue);
    //     if (particle1Index != particle1[i] || particle2Index != particle2[i])
    //         throw OpenMMException("updateParametersInContext: The set of particles in a bond has changed");
    //     length[i] = (RealOpenMM) lengthValue;
    //     kQuadratic[i] = (RealOpenMM) kValue;
    // }
}

RealOpenMM ReferenceCalcANN_ForceKernel::calculateForceAndEnergy(vector<RealVec>& positionData, vector<RealVec>& forceData) {
    // test case: add force on first atom, fix it at (0,0,0)
    RealOpenMM coef = 100.0;
    forceData[0][0]    -= coef * positionData[0][0];
    forceData[0][1]    -= coef * positionData[0][1];
    forceData[0][2]    -= coef * positionData[0][2];
    return 0;  // TODO: fix this later
}
