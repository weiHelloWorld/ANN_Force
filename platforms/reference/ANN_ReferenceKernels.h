#ifndef ANN_OPENMM_REFERENCE_KERNELS_H_
#define ANN_OPENMM_REFERENCE_KERNELS_H_


#include "openmm/System.h"
#include "openmm/ANN_Kernels.h"
#include "openmm/ANN_MultipoleForce.h"
#include "ANN_ReferenceMultipoleForce.h"
#include "ReferenceNeighborList.h"
#include "SimTKOpenMMRealType.h"

namespace OpenMM {

/**
 * This kernel is invoked by ANN_Force to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcANN_ForceKernel : public CalcANN_ForceKernel {
public:
    ReferenceCalcANN_ForceKernel(std::string name, 
                                               const Platform& platform,
                                               const System& system);
    ~ReferenceCalcANN_ForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the ANN_Force this kernel will be used for
     */
    void initialize(const System& system, const ANN_Force& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the ANN_Force to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const ANN_Force& force);
private:
    int nums;
    std::vector<int>   particle1;
    std::vector<int>   particle2;
    std::vector<RealOpenMM> length;
    std::vector<RealOpenMM> kQuadratic;
    RealOpenMM globalCubic;
    RealOpenMM globalQuartic;
    const System& system;
};


} // namespace OpenMM

#endif /*ANN_OPENMM_REFERENCE_KERNELS_H*/