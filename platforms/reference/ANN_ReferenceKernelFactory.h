#ifndef ANN_OPENMM_REFERERENCE_KERNEL_FACTORY_H_
#define ANN_OPENMM_REFERERENCE_KERNEL_FACTORY_H_


#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates all kernels for ANN_ReferencePlatform.
 */

class ANN_ReferenceKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*ANN_OPENMM_REFERERENCE_KERNEL_FACTORY_H_*/
