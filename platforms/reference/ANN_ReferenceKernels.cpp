#include "ANN_ReferenceKernels.h"
#include "openmm/internal/ANN_TorsionTorsionForceImpl.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/NonbondedForce.h"

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

ReferenceCalcANN_BondForceKernel::ReferenceCalcANN_BondForceKernel(std::string name, const Platform& platform, const System& system) : 
                CalcANN_BondForceKernel(name, platform), system(system) {
}

ReferenceCalcANN_BondForceKernel::~ReferenceCalcANN_BondForceKernel() {
}

void ReferenceCalcANN_BondForceKernel::initialize(const System& system, const ANN_BondForce& force) {

    numBonds = force.getNumBonds();
    for (int ii = 0; ii < numBonds; ii++) {

        int particle1Index, particle2Index;
        double lengthValue, kValue;
        force.getBondParameters(ii, particle1Index, particle2Index, lengthValue, kValue);

        particle1.push_back(particle1Index); 
        particle2.push_back(particle2Index); 
        length.push_back(static_cast<RealOpenMM>(lengthValue));
        kQuadratic.push_back(static_cast<RealOpenMM>(kValue));
    } 
    globalBondCubic   = static_cast<RealOpenMM>(force.getANN_GlobalBondCubic());
    globalBondQuartic = static_cast<RealOpenMM>(force.getANN_GlobalBondQuartic());
}

double ReferenceCalcANN_BondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& posData   = extractPositions(context);
    vector<RealVec>& forceData = extractForces(context);
    ANN_ReferenceBondForce ANN_ReferenceBondForce;
    RealOpenMM energy      = ANN_ReferenceBondForce.calculateForceAndEnergy(numBonds, posData, particle1, particle2, length, kQuadratic,
                                                                                       globalBondCubic, globalBondQuartic,
                                                                                       forceData);
    return static_cast<double>(energy);
}

void ReferenceCalcANN_BondForceKernel::copyParametersToContext(ContextImpl& context, const ANN_BondForce& force) {
    if (numBonds != force.getNumBonds())
        throw OpenMMException("updateParametersInContext: The number of bonds has changed");

    // Record the values.

    for (int i = 0; i < numBonds; ++i) {
        int particle1Index, particle2Index;
        double lengthValue, kValue;
        force.getBondParameters(i, particle1Index, particle2Index, lengthValue, kValue);
        if (particle1Index != particle1[i] || particle2Index != particle2[i])
            throw OpenMMException("updateParametersInContext: The set of particles in a bond has changed");
        length[i] = (RealOpenMM) lengthValue;
        kQuadratic[i] = (RealOpenMM) kValue;
    }
}

