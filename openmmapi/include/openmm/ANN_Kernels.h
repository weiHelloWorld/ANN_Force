#ifndef ANN_OPENMM_KERNELS_H_
#define ANN_OPENMM_KERNELS_H_


#include "openmm/KernelImpl.h"
#include "openmm/System.h"
#include "openmm/Platform.h"

#include <set>
#include <string>
#include <vector>

namespace OpenMM {

/**
 * This kernel is invoked by ANN_Force to calculate the forces acting on the system and the energy of the system.
 * it is a purely virtual class.
 */
class CalcANN_ForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcANN_Force";
    }

    CalcANN_ForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the ANN_Force this kernel will be used for
     */
    virtual void initialize(const System& system, const ANN_Force& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the ANN_Force to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const ANN_Force& force) = 0;
};



} // namespace OpenMM

#endif /*ANN_OPENMM_KERNELS_H*/
