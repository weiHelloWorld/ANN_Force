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

void print_matrix(double** matrix, int num_of_rows, int num_of_cols) {
    for (int ii = 0; ii < num_of_rows; ii ++) {
        for (int jj = 0; jj < num_of_cols; jj ++) {
            printf("%lf\t", matrix[ii][jj]);
        }
        printf("\n");
    }
    return;
}

void print_vector(vector<double> vec, int num) {
    for (int ii = 0; ii < num; ii ++) {
        printf("%lf\t", vec[ii]);
    }
    printf("\n");
    return;
}

void print_Vec3(Vec3 temp_vec) {
    for (int ii = 0; ii < 3; ii ++) {
        printf("%f\t", temp_vec[ii]);
    }
    printf("\n");
    return;
}

void test_1() {
    cout << "running test_1\n";
    System system;
    int num_of_atoms = 4;
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        system.addParticle(1.0);    
    }
    VerletIntegrator integrator(0.01);
    ANN_Force* forceField = new ANN_Force();
    system.addForce(forceField);
    Platform& platform = Platform::getPlatformByName("Reference");
    // cout << "platform = " << platform.getName() << endl;
    Context context(system, integrator, platform);
    vector<Vec3> positions(num_of_atoms);
    positions[0] = Vec3(-1, -2, -3);
    positions[1] = Vec3(0, 0, 0);
    positions[2] = Vec3(1, 0, 0);
    positions[3] = Vec3(0, 0, 1);
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
    cout << "running test_sincos_of_dihedrals_four_atom\n";
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


void test_forward_and_backward_prop() {
    cout << "running test_forward_and_backward_prop\n";
    System system;
    ANN_Force* forceField = new ANN_Force();
    Platform& platform = Platform::getPlatformByName("Reference");
    ReferenceCalcANN_ForceKernel* forcekernel = new ReferenceCalcANN_ForceKernel("", platform, system);
    forceField -> set_num_of_nodes(vector<int>({2, 3, 1}));
    vector<vector<double> > coeff{{0.01,0.02,0.03,0.04,0.05,0.06}, {-1, -2, -3}};
    forceField -> set_coeffients_of_connections(coeff);
    vector<vector<double> > bias{{0.01,0.02,0.03},{2}};
    forceField -> set_values_of_biased_nodes(bias);
    vector<string> layer_types {"Linear", "Tanh"};
    forceField -> set_layer_types(layer_types);
    vector<double> pc{0};
    forceField -> set_potential_center(pc);
    forceField -> set_force_constant(10);
    forcekernel -> initialize(system, *forceField);
    auto temp_coef = forcekernel -> get_coeff();
    print_matrix(temp_coef[0], 3, 2);
    print_matrix(temp_coef[1], 1, 3);
    vector<RealOpenMM> input{1,2};
    forcekernel -> calculate_output_of_each_layer(input);
    auto actual_output_of_layer = forcekernel -> get_output_of_each_layer();
    vector<vector<double> > expected_output_of_layer{{1,2}, {0.06,0.13,0.20},{tanh(1.08)}};
    for (int ii = 0; ii < NUM_OF_LAYERS; ii ++) {
        for (int jj = 0; jj < expected_output_of_layer[ii].size(); jj ++) {
            ASSERT_EQUAL_TOL(expected_output_of_layer[ii][jj], actual_output_of_layer[ii][jj], TOL);
        }
    }
    vector<vector<double> > result;
    forcekernel -> back_prop(result);
    print_vector(result[0], 2);
    print_vector(result[1], 3);
    print_vector(result[2], 1);
    return;
}

void test_calculation_of_forces_by_comparing_with_numerical_derivatives() {
    cout << "running test_calculation_of_forces_by_comparing_with_numerical_derivatives\n";
    System system;
    int num_of_atoms = 6;
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        system.addParticle(1.0);    
    }
    VerletIntegrator integrator(0.01);
    ANN_Force* forceField = new ANN_Force();
    forceField -> set_num_of_nodes(vector<int>({4, 4, 4}));
    forceField -> set_layer_types(vector<string>({"Linear", "Tanh"}));
    vector<vector<double> > coeff{{1, 0, 0, 0,
                                   0, 1, 0, 0,
                                   0, 0, 1, 0,
                                   0, 0, 0, 1
                                    }, 
                                  {1, 0, 0, 0,
                                   0, 1, 0, 0,
                                   0, 0, 1, 0,
                                   0, 0, 0, 1
                                    }};
    forceField -> set_coeffients_of_connections(coeff);
    forceField -> set_force_constant(10);
    forceField -> set_potential_center(vector<double>({0, 0, 0, 0}));
    forceField -> set_values_of_biased_nodes(vector<vector<double> > {{0}, {0}});
    forceField -> set_index_of_backbone_atoms(vector<int>({0, 1, 2, 3, 4, 5}));
    system.addForce(forceField);
    Platform& platform = Platform::getPlatformByName("Reference");
    Context context(system, integrator, platform);
    vector<Vec3> positions_1(num_of_atoms);
    positions_1[0] = Vec3(-1, -2, -3);
    positions_1[1] = Vec3(0, 0, 0);
    positions_1[2] = Vec3(1, 0, 0);
    positions_1[3] = Vec3(0, 0, 1);
    positions_1[4] = Vec3(0.5, 0, 0);
    positions_1[5] = Vec3(0, 0.3, 0.6);
    context.setPositions(positions_1);

    double energy_1, energy_2, energy_3;

    vector<Vec3> forces;
    vector<Vec3> temp_positions;

    State state = context.getState(State::Forces | State::Energy | State::Positions);
    {
        forces = state.getForces();
        energy_1 = state.getPotentialEnergy();
        temp_positions = state.getPositions();
        printf("forces:\n");
        for (int ii = 0; ii < num_of_atoms; ii ++) {
            print_Vec3(forces[ii]);
        }
        printf("positions:\n");
        for (int ii = 0; ii < num_of_atoms; ii ++) {
            print_Vec3(temp_positions[ii]);
        }
        printf("potential energy = %lf\n", energy_1);
    }

    double delta = 0.02;
    auto positions_2 = positions_1;
    auto numerical_derivatives = forces; // we need to compare this numerical result with the forces calculated
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        for (int jj = 0; jj < 3; jj ++) {
            positions_2 = positions_1;
            positions_2[ii][jj] += delta;
            context.setPositions(positions_2);
            energy_2 = context.getState(State::Forces | State::Energy | State::Positions).getPotentialEnergy();
            // printf("potential energy = %lf\n", energy_2);
            numerical_derivatives[ii][jj] = (energy_2 - energy_1) / delta;
        }
    }
    // print out numerical results
    printf("numerical_derivatives = \n");
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        print_Vec3(numerical_derivatives[ii]);
    }
    
    return;
}



int main(int argc, char* argv[]) {
    try {
        registerKernelFactories();  // this is required
        // test_1();
        // test_sincos_of_dihedrals_four_atom();
        // test_forward_and_backward_prop();
        test_calculation_of_forces_by_comparing_with_numerical_derivatives();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}