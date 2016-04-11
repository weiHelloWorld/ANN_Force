#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/Platform.h"

#include <iostream>
#include <vector>
#include <stdio.h>

#include "../../platforms/reference/ANN_ReferenceKernelFactory.cpp"

using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;

void test_1() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    ANN_Force* forceField = new ANN_Force();
    system.addForce(forceField);
    Platform& platform = Platform::getPlatformByName("Reference");
    // cout << "platform = " << platform.getName() << endl;
    Context context(system, integrator, platform);
    vector<Vec3> positions(3);
    positions[0] = Vec3(-1, -2, -3);
    positions[1] = Vec3(0, 0, 0);
    positions[2] = Vec3(1, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    {
        const vector<Vec3>& forces = state.getForces();
        ASSERT_EQUAL_VEC(Vec3(110.0, 220.0, 300.0), forces[0], TOL);
        ASSERT_EQUAL_VEC(Vec3(0, 0, 0), forces[1], TOL);
        ASSERT_EQUAL_VEC(Vec3(0, 0, 0), forces[2], TOL);
    }
}

void test_sincos_of_dihedrals_four_atom() {
    System system;
    Platform& platform = Platform::getPlatformByName("Reference");
    ReferenceCalcANN_ForceKernel* forcekernel = new ReferenceCalcANN_ForceKernel("", platform, system);
    vector<RealVec> positionData(4);
    positionData[0] = Vec3(0, 1, 0);
    positionData[1] = Vec3(0, 0, 0);
    positionData[2] = Vec3(1, 0, 0);
    positionData[3] = Vec3(0, 0, 1);
    RealOpenMM cos_value, sin_value;
    forcekernel -> get_cos_and_sin_for_four_atoms(0,1,2,3, positionData, cos_value, sin_value);
    ASSERT_EQUAL_TOL(cos_value, 0, TOL);
    ASSERT_EQUAL_TOL(sin_value, -1, TOL);
    return;
}


void test_2() {
    System system;
    ANN_Force* forceField = new ANN_Force();
    Platform& platform = Platform::getPlatformByName("Reference");
    ReferenceCalcANN_ForceKernel* forcekernel = new ReferenceCalcANN_ForceKernel("", platform, system);
    vector<int> num_of_nodes({2, 3, 1});
    forceField -> set_num_of_nodes(num_of_nodes);
    vector<vector<double> > coeff{{1,2,3,4,5,6}, {-1, -2, -3}};
    forceField -> set_coeffients_of_connections(coeff);
    forcekernel -> initialize(system, *forceField);
    auto temp_coef = forcekernel -> get_coeff();
    print_matrix(temp_coef[0], 2, 3);
    print_matrix(temp_coef[1], 3, 1);
    return;
}


int main(int argc, char* argv[]) {
    try {
        registerKernelFactories();  // this is required
        test_1();
        test_sincos_of_dihedrals_four_atom();
        test_2();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}