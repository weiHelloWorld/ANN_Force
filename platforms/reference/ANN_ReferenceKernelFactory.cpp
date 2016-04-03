
#include "ANN_ReferenceKernelFactory.h"
#include "ANN_ReferenceKernels.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
             ANN_ReferenceKernelFactory* factory = new ANN_ReferenceKernelFactory();
             platform.registerKernelFactory(CalcANN_ForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerANN_ReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ANN_ReferenceKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& referencePlatformData = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());


    if (name == CalcANN_ForceKernel::Name())
        return new ReferenceCalcANN_ForceKernel(name, platform, context.getSystem());


    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
