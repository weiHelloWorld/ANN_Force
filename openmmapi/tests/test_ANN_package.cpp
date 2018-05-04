#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/Platform.h"
#include "OpenMM_ANN.h"
#include "../../platforms/reference/ANN_ReferenceKernelFactory.h"
#include "../../platforms/reference/ANN_ReferenceKernels.h"
#include <fstream>
#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <vector>
#include <stdio.h>
#include <time.h>
using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;
const double FORCE_TOL = 5e-2;
// #define PRINT_FORCE_1

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
    // this test is for candidate_1()
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
    // print_matrix(temp_coef[0], 3, 2);
    // print_matrix(temp_coef[1], 1, 3);
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
    // print_vector(result[0], 2);
    // print_vector(result[1], 3);
    // print_vector(result[2], 1);
    return;
}

void test_forward_and_backward_prop_2() {
    cout << "running test_forward_and_backward_prop_2\n";
    System system;
    ANN_Force* forceField = new ANN_Force();
    Platform& platform = Platform::getPlatformByName("Reference");
    ReferenceCalcANN_ForceKernel* forcekernel = new ReferenceCalcANN_ForceKernel("", platform, system);
    forceField -> set_num_of_nodes(vector<int>({2, 3, 2}));
    vector<vector<double> > coeff{{0.01,0.02,0.03,0.04,0.05,0.06}, {-1, -2, -3, 1, 2, 3}};
    forceField -> set_coeffients_of_connections(coeff);
    vector<vector<double> > bias{{0.01,0.02,0.03},{1.92, 1.08}};
    forceField -> set_values_of_biased_nodes(bias);
    vector<string> layer_types {"Linear", "Circular"};
    forceField -> set_layer_types(layer_types);
    vector<double> pc{0};
    forceField -> set_potential_center(pc);
    forceField -> set_force_constant(10);
    forcekernel -> initialize(system, *forceField);
    auto temp_coef = forcekernel -> get_coeff();
    vector<RealOpenMM> input{1,2};
    forcekernel -> calculate_output_of_each_layer(input);
    auto actual_output_of_layer = forcekernel -> get_output_of_each_layer();
    vector<vector<double> > expected_output_of_layer{{1,2}, {0.06,0.13,0.20},{0.447214, 0.894427}};
    for (int ii = 0; ii < NUM_OF_LAYERS; ii ++) {
        printf("actual_output_of_layer[%d] = \n", ii);
        print_vector(actual_output_of_layer[ii], actual_output_of_layer[ii].size());
    }
    
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

void assert_forces_equal_derivatives(vector<Vec3> forces, vector<Vec3> numerical_derivatives) {
    for (int ii = 0; ii < forces.size(); ii ++) {
        ASSERT_EQUAL_VEC(forces[ii], - numerical_derivatives[ii], FORCE_TOL);
    }
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
    forceField -> set_layer_types(vector<string>({"Tanh", "Tanh"}));
    vector<vector<double> > coeff{{1, 0, 0, 0,
                                   0, 1, 0, 0.7,
                                   0, 0, 1, 0,
                                   0, 0, 0, 1
                                    }, 
                                  {1, 0, 0.4, 0,
                                   0, 1, 0, 0,
                                   0, 0, 1, 0,
                                   0, 0, 0, 1
                                    }};
    forceField -> set_coeffients_of_connections(coeff);
    forceField -> set_force_constant(10);
    forceField -> set_potential_center(vector<double>({0, 0, 0, 0}));
    forceField -> set_values_of_biased_nodes(vector<vector<double> > {{0.1,0.2,0.3,0.4}, {0.5,0.6,0.4,0.3}});
    forceField -> set_list_of_index_of_atoms_forming_dihedrals_from_index_of_backbone_atoms(vector<int>({1, 2, 3, 4, 5, 6}));
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
#ifdef PRINT_FORCE
        printf("forces:\n");
        for (int ii = 0; ii < num_of_atoms; ii ++) {
            print_Vec3(forces[ii]);
        }
#endif
    }

    double delta = 0.005;
    auto positions_2 = positions_1;
    auto numerical_derivatives = forces; // we need to compare this numerical result with the forces calculated
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        for (int jj = 0; jj < 3; jj ++) {
            positions_2 = positions_1;
            positions_2[ii][jj] += delta;
            context.setPositions(positions_2);
            energy_2 = context.getState(State::Energy | State::Positions).getPotentialEnergy();
            numerical_derivatives[ii][jj] = (energy_2 - energy_1) / delta;
        }
    }
    // print out numerical results
#ifdef PRINT_FORCE
    printf("numerical_derivatives = \n");
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        print_Vec3(numerical_derivatives[ii]);
    }
#endif
    assert_forces_equal_derivatives(forces, numerical_derivatives);
    return;
}

void test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_circular_layer(vector<double> potential_center) {
    cout << "running test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_circular_layer, with potential_center = [" 
            << potential_center[0] << "," << potential_center[1] << "]\n";
    System system;
    int num_of_atoms = 6;
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        system.addParticle(1.0);    
    }
    VerletIntegrator integrator(0.01);
    ANN_Force* forceField = new ANN_Force();
    forceField -> set_num_of_nodes(vector<int>({4, 4, 4}));
    forceField -> set_layer_types(vector<string>({"Tanh", "Circular"}));
    vector<vector<double> > coeff{{1, 0, 0, 0,
                                   0, 1, 0, 0.7,
                                   0, 0, 1, 0,
                                   0, 0, 0, 1
                                    }, 
                                  {1, 0, 0.4, 0,
                                   0, 1, 0, 0,
                                   0, 0, 1, 0,
                                   0, 0, 0, 1
                                    }};
    forceField -> set_coeffients_of_connections(coeff);
    forceField -> set_force_constant(10);
    forceField -> set_potential_center(potential_center);
    forceField -> set_values_of_biased_nodes(vector<vector<double> > {{0.1,0.2,0.3,0.4}, {0.5,0.6,0.4,0.3}});
    forceField -> set_list_of_index_of_atoms_forming_dihedrals_from_index_of_backbone_atoms(vector<int>({1, 2, 3, 4, 5, 6}));
    cout << "data_type_in_input_layer = " << forceField -> get_data_type_in_input_layer() << endl;
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
#ifdef PRINT_FORCE
        printf("forces:\n");
        for (int ii = 0; ii < num_of_atoms; ii ++) {
            print_Vec3(forces[ii]);
        }
#endif
    }

    double delta = 0.005;
    auto positions_2 = positions_1;
    auto numerical_derivatives = forces; // we need to compare this numerical result with the forces calculated
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        for (int jj = 0; jj < 3; jj ++) {
            positions_2 = positions_1;
            positions_2[ii][jj] += delta;
            context.setPositions(positions_2);
            energy_2 = context.getState(State::Forces | State::Energy | State::Positions).getPotentialEnergy();
            numerical_derivatives[ii][jj] = (energy_2 - energy_1) / delta;
        }
    }
    // print out numerical results
#ifdef PRINT_FORCE
    printf("numerical_derivatives = \n");
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        print_Vec3(numerical_derivatives[ii]);
    }
#endif
    assert_forces_equal_derivatives(forces, numerical_derivatives);

    return;
}

void test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_alanine_dipeptide() {
    // this is a test for alanine dipeptide (only backbone atoms included here)
    cout << "running test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_alanine_dipeptide\n";
    System system;
    int num_of_atoms = 7;
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        system.addParticle(1.0);    
    }
    VerletIntegrator integrator(0.01);
    ANN_Force* forceField = new ANN_Force();
    forceField -> set_num_of_nodes(vector<int>({8, 15, 2}));
    forceField -> set_layer_types(vector<string>({"Tanh", "Tanh"}));
    vector<vector<double> > coeff{{-0.31875740151672177, 0.32285065440356292, 0.83635626972959354, 0.14920100171609049, 0.41869803015016493, 0.53283392495651005, 0.25247090132182243, -0.61701125566675374, -0.22607647879242615, -0.24483169515428535, -0.1013238998466106, -0.077129877014225343, 0.37501725915420098, -0.2294719896347639, 0.1590300683353377, -0.10538882715479421, 0.0026284642033629645, -0.30266922459815626, -0.45797543631052889, 0.7007463842530699, 0.047775242961448312, -0.78382992805123353, -0.4660494071930491, -0.39201128983194611, 0.24235126560716347, -0.015758468776178744, -0.13270024896630694, 0.096479474684857264, 0.27607049730326139, 0.065758748600758324, -0.28279839081073049, 0.27599697815723212, -0.70262619251399183, 0.48656474098699148, 0.82135309047190264, -0.050361428599173044, 0.24560225793804594, -0.079517500243165409, -0.23796263769471743, 0.85009183566475033, 0.087365375302592099, -0.052921401392980856, -0.29817084048926168, -1.0326386472698428, 0.20342980179953343, -0.13558239533849584, 0.1957756940735923, 0.67328683258697686, 0.32441309795557471, 0.3195945168934492, -0.86778633811838624, 0.91485999348761271, -0.25170021443870172, 0.24673389983866131, 0.029137312303548438, -0.5210910731063183, -0.56035387773509293, 0.65786248121684121, 0.17997997257016271, 0.014974474870441058, 0.47449115820104976, 0.007350748851190708, 0.31473027663345465, 0.65601331047445666, -0.44739702747442728, -0.16316909854712905, 0.56031920409035618, -0.55875349389516837, -0.18124762143230999, -0.099381036437795611, 0.099402112600022574, -0.022793146545307737, -0.60091578790246902, -0.4138218704118381, 0.34204371061533623, 0.070605431381411871, 0.2394063271441117, -0.59311375567961644, 0.3401878060194638, -0.12007832877693034, 0.39437682918237271, -0.21771192199710357, -0.46956852466628901, -0.56832621063843392, 0.43720719391065177, 0.2411460299507433, 0.62170713890364371, -0.66051006858886507, 0.57036194295819731, 0.63727608398652691, -0.53224693296258119, -0.099093053473522708, 0.63659918168912422, 0.58107499352913794, 0.47122345020639395, 0.095072382502705111, -0.071411494136139828, 0.32460289192742814, -0.34704776814802485, 0.16984524543900381, -0.50136780370991874, 0.64155385260634823, 0.16226407797460771, 0.2077508829749542, 0.4892371719552745, 0.39553852725674848, -0.44811960854132266, -0.23242246681172982, 0.30704103629042717, 0.37729321226518886, 0.13259358612305974, 0.204788284194278, -0.59652605867517583, 0.63760108464365683, 0.38997085110406143, 0.54243902741026873, -0.46422585669009203, -0.48585444096773639, -0.18920213683108078, -0.062503740975689823},
{0.35052979094807973, -0.031435892979307982, -0.34116898152427017, 0.1941037639473189, 0.37354211293956591, 0.5614821322429786, 0.24316885912756434, -0.03130579471899831, -0.086238937592167039, -0.56507487379601962, 0.0167584352903551, -0.33610604695343876, -0.10152981934629646, -0.96251698105359274, 0.24009067471188655, 0.2844259997867663, 0.71001035426183579, -0.40100233063691015, -0.065801436146621969, -0.32518888534882978, 0.094252392979903885, -0.17129984904380494, 0.44300308066693594, 0.20601112070470307, 0.08278971723751706, -0.12741076093291512, -0.11315603552563522, -0.41884452991601434, 0.049954057246054118, -0.33594067654411891}};
    forceField -> set_coeffients_of_connections(coeff);
    forceField -> set_force_constant(10);
    forceField -> set_potential_center(vector<double>({0.6, 0.5}));
    forceField -> set_values_of_biased_nodes(vector<vector<double> > 
        {{-0.36906062280709279, -0.59131919312399095, -0.84659856651584509, -0.084765966504940352, 0.47258669905786538, 0.29139864407550331, 0.039776531427744038, -0.12426019578607678, 0.27813782873304532, 0.12349188418434202, -0.23928032048694292, -0.15582822433851695, 0.71783542899694119, 0.13163101710811542, 0.14141529504476155},
{.090182463663168511, 0.58352910786332224}});
    forceField -> set_list_of_index_of_atoms_forming_dihedrals(vector<vector<int> >{{1,2,3,4}, 
                                                                                    {2,3,4,5},
                                                                                    {3,4,5,6},
                                                                                    {4,5,6,7}});
    system.addForce(forceField);
    Platform& platform = Platform::getPlatformByName("Reference");
    Context context(system, integrator, platform);
    vector<Vec3> positions_1(num_of_atoms);
    positions_1[0] = Vec3(-1, -2, -3);
    positions_1[1] = Vec3(0, 0, 0);
    positions_1[2] = Vec3(1.5, 0, 0);
    positions_1[3] = Vec3(0, 0.57, 1);
    positions_1[4] = Vec3(0.5, 0, 0.1);
    positions_1[5] = Vec3(0, 0.3, 0.6);
    positions_1[6] = Vec3(0, 0.4, 0.5);
    context.setPositions(positions_1);

    double energy_1, energy_2, energy_3;

    vector<Vec3> forces;
    vector<Vec3> temp_positions;

    State state = context.getState(State::Forces | State::Energy | State::Positions);
    {
        forces = state.getForces();
        energy_1 = state.getPotentialEnergy();
        temp_positions = state.getPositions();
#ifdef PRINT_FORCE
        printf("forces:\n");
        for (int ii = 0; ii < num_of_atoms; ii ++) {
            print_Vec3(forces[ii]);
        }
#endif
    }

    double delta = 0.005;
    auto positions_2 = positions_1;
    auto numerical_derivatives = forces; // we need to compare this numerical result with the forces calculated
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        for (int jj = 0; jj < 3; jj ++) {
            positions_2 = positions_1;
            positions_2[ii][jj] += delta;
            context.setPositions(positions_2);
            energy_2 = context.getState(State::Energy | State::Positions).getPotentialEnergy();
            numerical_derivatives[ii][jj] = (energy_2 - energy_1) / delta;
        }
    }
    // print out numerical results
#ifdef PRINT_FORCE
    printf("numerical_derivatives = \n");
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        print_Vec3(numerical_derivatives[ii]);
    }
#endif
    assert_forces_equal_derivatives(forces, numerical_derivatives);
    
    return;
}

void test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_Cartesian_coordinates(string temp_platform) {
    cout << "running test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_Cartesian_coordinates ";
    cout << "(" << temp_platform << ")\n";
    System system;
    int num_of_atoms = 4;
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        system.addParticle(1.0);    
    }
    VerletIntegrator integrator(0.01);
    ANN_Force* forceField = new ANN_Force();
    forceField -> set_num_of_nodes(vector<int>({12, 4, 4}));
    forceField -> set_layer_types(vector<string>({"Tanh", "Tanh"}));
    vector<vector<double> > coeff{{1,1,1,0,0,0,0,0,0,0,0,0,
                                   0,0,0,1,1,1,0,0,0,0,0,0,
                                   0,0,0,0,0,0,1,1,1,0,0,0,
                                   0,0,0,0,0,0,0,0,0,1,1,1
                                    }, 
                                  {1, 0, 0.4, 0,
                                   0, 1, 0, 0,
                                   0, 0, 1, 0,
                                   0, 0, 0, 1
                                    }};
    forceField -> set_coeffients_of_connections(coeff);
    forceField -> set_force_constant(10);
    forceField -> set_scaling_factor(1);
    forceField -> set_index_of_backbone_atoms({1,2,3,4});
    forceField -> set_potential_center(vector<double>({0, 0, 0, 0}));
    forceField -> set_values_of_biased_nodes(vector<vector<double> > {{0.1,0.2,0.3,0.4}, {0.5,0.6,0.4,0.3}});
    forceField -> set_data_type_in_input_layer(1);
    system.addForce(forceField);
    Platform& platform = Platform::getPlatformByName(temp_platform);
    Context context(system, integrator, platform);
    vector<Vec3> positions_1(num_of_atoms);
    positions_1[0] = Vec3(-1, -2, -3);
    positions_1[1] = Vec3(0, 0, 0);
    positions_1[2] = Vec3(1, 0, 0);
    positions_1[3] = Vec3(0, 0, 2);
    context.setPositions(positions_1);

    double energy_1, energy_2, energy_3;

    vector<Vec3> forces;
    vector<Vec3> temp_positions;

    State state = context.getState(State::Forces | State::Energy | State::Positions);
    {
        forces = state.getForces();
        energy_1 = state.getPotentialEnergy();
        temp_positions = state.getPositions();
#ifdef PRINT_FORCE
        printf("forces:\n");
        for (int ii = 0; ii < num_of_atoms; ii ++) {
            print_Vec3(forces[ii]);
        }
#endif
    }

    double delta = 0.005;
    auto positions_2 = positions_1;
    auto numerical_derivatives = forces; // we need to compare this numerical result with the forces calculated
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        for (int jj = 0; jj < 3; jj ++) {
            positions_2 = positions_1;
            positions_2[ii][jj] += delta;
            context.setPositions(positions_2);
            energy_2 = context.getState(State::Energy | State::Positions).getPotentialEnergy();
            numerical_derivatives[ii][jj] = (energy_2 - energy_1) / delta;
        }
    }
    // print out numerical results
#ifdef PRINT_FORCE
    printf("numerical_derivatives = \n");
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        print_Vec3(numerical_derivatives[ii]);
    }
#endif
    assert_forces_equal_derivatives(forces, numerical_derivatives);
    
    return;
}

vector<double> generate_random_array(int size, int seed) {
    srand((unsigned)seed);
    vector<double> temp_array;
    for(int i = 0; i < size; i ++){ 
        temp_array.push_back( (rand() % 10) / 100.0);
    }
    return temp_array;
}

void test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_Cartesian_coordinates_larger_system(
    string temp_platform, int num_of_atoms, int seed) {
    cout << "running test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_Cartesian_coordinates_larger_system ";
    cout << "(platform = " << temp_platform << ", num_of_atoms = " << num_of_atoms << ")\n";
    time_t start_timer, end_timer;
    double runtime_seconds;
    time(&start_timer);
    System system;
    vector<int> index_of_backbone_atoms(num_of_atoms);
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        system.addParticle(1.0);   
        index_of_backbone_atoms[ii] = ii + 1; 
    }
    VerletIntegrator integrator(0.01);
    ANN_Force* forceField = new ANN_Force();
    int num_nodes_first_layer = 3 * num_of_atoms;
    forceField -> set_num_of_nodes(vector<int>({num_nodes_first_layer, 150, 4}));
    forceField -> set_layer_types(vector<string>({"Tanh", "Tanh"}));
    vector<vector<double> > coeff;
    coeff.push_back(generate_random_array(num_nodes_first_layer * 150, seed));
    coeff.push_back(generate_random_array(150 * 4, seed));
    forceField -> set_coeffients_of_connections(coeff);
    forceField -> set_force_constant(10);
    forceField -> set_scaling_factor(1);
    forceField -> set_index_of_backbone_atoms(index_of_backbone_atoms);
    forceField -> set_potential_center(vector<double>({0, 0, 0, 0}));
    vector<vector<double> > biased_notes_vec;
    biased_notes_vec.push_back(generate_random_array(150, seed));
    biased_notes_vec.push_back(generate_random_array(4, seed));
    forceField -> set_values_of_biased_nodes(biased_notes_vec);
    forceField -> set_data_type_in_input_layer(1);
    system.addForce(forceField);
    Platform& platform = Platform::getPlatformByName(temp_platform);
    Context context(system, integrator, platform);
    vector<Vec3> positions_1(num_of_atoms);
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        positions_1[ii] = Vec3((rand() % 100) / 20.0, (rand() % 100) / 20.0, (rand() % 100) / 20.0);
    }
    context.setPositions(positions_1);

    double energy_1, energy_2, energy_3;

    vector<Vec3> forces;
    vector<Vec3> temp_positions;

    State state = context.getState(State::Forces | State::Energy | State::Positions);
    {
        forces = state.getForces();
        energy_1 = state.getPotentialEnergy();
        temp_positions = state.getPositions();
#ifdef PRINT_FORCE
        printf("forces:\n");
        for (int ii = 0; ii < num_of_atoms; ii ++) {
            print_Vec3(forces[ii]);
        }
#endif
    }
    double delta = 0.005;
    auto positions_2 = positions_1;
    auto numerical_derivatives = forces; // we need to compare this numerical result with the forces calculated
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        for (int jj = 0; jj < 3; jj ++) {
            positions_2 = positions_1;
            positions_2[ii][jj] += delta;
            context.setPositions(positions_2);
            energy_2 = context.getState(State::Energy | State::Positions).getPotentialEnergy();
            numerical_derivatives[ii][jj] = (energy_2 - energy_1) / delta;
        }
    }
    // print out numerical results
#ifdef PRINT_FORCE
    printf("numerical_derivatives = \n");
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        print_Vec3(numerical_derivatives[ii]);
    }
#endif
    assert_forces_equal_derivatives(forces, numerical_derivatives);
    time(&end_timer);
    runtime_seconds = difftime(end_timer, start_timer);
    printf("running time = %f\n", runtime_seconds);
    return;
}

void test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_pairwise_distances(string temp_platform) {
    cout << "running test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_pairwise_distances ";
    cout << "(" << temp_platform << ")\n";
    System system;
    int num_of_atoms = 4;
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        system.addParticle(1.0);    
    }
    VerletIntegrator integrator(0.01);
    ANN_Force* forceField = new ANN_Force();
    forceField -> set_index_of_backbone_atoms({1,2,3,4});
    forceField -> set_num_of_nodes(vector<int>({6, 3, 3}));
    forceField -> set_layer_types(vector<string>({"Tanh", "Tanh"}));
    forceField -> set_list_of_pair_index_for_distances(vector<vector<int> >({{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}}));
    vector<vector<double> > coeff{{1,1,0,0,0,0,
                                   0,0,1,1,0,0.7,
                                   0,0,0,0,1,1
                                    }, 
                                  {1, 0, 0.4,
                                   0, 1, 0,
                                   0, 0, 1,
                                    }};
    forceField -> set_coeffients_of_connections(coeff);
    forceField -> set_force_constant(10);
    forceField -> set_scaling_factor(10);
    forceField -> set_potential_center(vector<double>({0, 0, 0}));
    forceField -> set_values_of_biased_nodes(vector<vector<double> > {{0.1,0.2,0.3}, {0.5,0.6,0.4}});
    forceField -> set_data_type_in_input_layer(2); // pairwise distances
    system.addForce(forceField);
    Platform& platform = Platform::getPlatformByName(temp_platform);
    Context context(system, integrator, platform);
    vector<Vec3> positions_1(num_of_atoms);
    positions_1[0] = Vec3(-1, -2, -3);
    positions_1[1] = Vec3(0, 0, 0);
    positions_1[2] = Vec3(1, 0, 0);
    positions_1[3] = Vec3(0, 0, 2);
    context.setPositions(positions_1);

    double energy_1, energy_2, energy_3;

    vector<Vec3> forces;
    vector<Vec3> temp_positions;

    State state = context.getState(State::Forces | State::Energy | State::Positions);
    {
        forces = state.getForces();
        energy_1 = state.getPotentialEnergy();
        temp_positions = state.getPositions();
#ifdef PRINT_FORCE_1
        printf("forces:\n");
        for (int ii = 0; ii < num_of_atoms; ii ++) {
            print_Vec3(forces[ii]);
        }
#endif
    }

    double delta = 0.005;
    auto positions_2 = positions_1;
    auto numerical_derivatives = forces; // we need to compare this numerical result with the forces calculated
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        for (int jj = 0; jj < 3; jj ++) {
            positions_2 = positions_1;
            positions_2[ii][jj] += delta;
            context.setPositions(positions_2);
            energy_2 = context.getState(State::Energy | State::Positions).getPotentialEnergy();
            numerical_derivatives[ii][jj] = (energy_2 - energy_1) / delta;
        }
    }
    // print out numerical results
#ifdef PRINT_FORCE
    printf("numerical_derivatives = \n");
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        print_Vec3(numerical_derivatives[ii]);
    }
#endif
    assert_forces_equal_derivatives(forces, numerical_derivatives);
    
    return;
}

int main(int argc, char* argv[]) {
    try {
        OpenMM::Platform::loadPluginsFromDirectory("/home/kengyangyao/.openmm/lib/plugins");
        test_forward_and_backward_prop();
        test_forward_and_backward_prop_2();
        test_calculation_of_forces_by_comparing_with_numerical_derivatives();
        test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_circular_layer(vector<double>({0, 0}));
        test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_circular_layer(vector<double>({2.4, 2.3}));
        test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_alanine_dipeptide();
        test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_Cartesian_coordinates("Reference");
        test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_Cartesian_coordinates("CUDA");
        test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_pairwise_distances("Reference");
        test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_pairwise_distances("CUDA");
        for (int num_of_atoms = 20; num_of_atoms < 200; num_of_atoms += 20) {
            test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_Cartesian_coordinates_larger_system(
                "Reference", num_of_atoms, 1);
            test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_Cartesian_coordinates_larger_system(
                "CUDA", num_of_atoms, 1);
        }
        // test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_Cartesian_coordinates_larger_system(
        //         "CUDA", 500, 1);
        
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}