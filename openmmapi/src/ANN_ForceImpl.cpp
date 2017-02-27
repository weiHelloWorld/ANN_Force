#include "openmm/internal/ContextImpl.h"
#include "openmm/Platform.h"
#include <set>
#include "openmm/internal/ANN_ForceImpl.h"
#include "openmm/ANN_Kernels.h"
#include "openmm/OpenMMException.h"
#include <cmath>

using namespace OpenMM;

using std::pair;
using std::vector;
using std::set;

ANN_ForceImpl::ANN_ForceImpl(const ANN_Force& owner) : owner(owner) {
}

ANN_ForceImpl::~ANN_ForceImpl() {
}

void ANN_ForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcANN_ForceKernel::Name(), context);
    kernel.getAs<CalcANN_ForceKernel>().initialize(context.getSystem(), owner);
}

double ANN_ForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcANN_ForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> ANN_ForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcANN_ForceKernel::Name());
    return names;
}


void ANN_ForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcANN_ForceKernel>().copyParametersToContext(context, owner);
}
