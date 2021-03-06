#include "OpenMM.h"
#include <cstdio>
#include "openmm/ANN_Force.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/PluginInitializer.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

// extern "C" OPENMM_EXPORT void registerExampleReferenceKernelFactories();

// Forward declaration of routine for printing one frame of the
// trajectory, defined later in this source file.
void writePdbFrame(int frameNum, const OpenMM::State&);

void print_Vec3(Vec3 temp_vec) {
    for (int ii = 0; ii < 3; ii ++) {
        printf("%f\t", temp_vec[ii]);
    }
    printf("\n");
    return;
}

void test(string mode)
{
    // Load any shared libraries containing GPU implementations.
    // OpenMM::Platform::loadPluginsFromDirectory(
    //     OpenMM::Platform::getDefaultPluginsDirectory());

    // Create a system with nonbonded forces.
    System system;
    int num_of_atoms = 5;
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        system.addParticle(1.0);    
    }
    VerletIntegrator integrator(0.01);
    ANN_Force* forceField = new ANN_Force();
    forceField -> set_num_of_nodes(vector<int>({15, 5, 4}));
    forceField -> set_layer_types(vector<string>({"Tanh", "Tanh"}));
    vector<vector<double> > coeff{{1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
                                   0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,
                                   0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
                                   0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,
                                   0,0,0,0,0,0,0,0,0,0,0,0,1,1,1
                                    }, 
                                  {1, 0, 0.4, 0, 0,
                                   0, 1, 0, 0, 0,
                                   0, 0, 1, 0, 0,
                                   0, 0, 0, 1, 0,
                                   //0, 0, 0, 0, 1
                                    }};
    forceField -> set_coeffients_of_connections(coeff);
    forceField -> set_force_constant(10);
    forceField -> set_scaling_factor(1);
    forceField -> set_index_of_backbone_atoms({1,2,3,4,5});
    forceField -> set_potential_center(vector<double>({0, 0, 0,0}));
    forceField -> set_values_of_biased_nodes(vector<vector<double> > {{0.1,0.2,0.3,0.4}, {0.5,0.6,0.4,0.3}});
    forceField -> set_data_type_in_input_layer(1);
    cout << "data_type_in_input_layer = " << forceField -> get_data_type_in_input_layer() << endl;
    system.addForce(forceField);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integrator, platform);
    vector<Vec3> positions_1(num_of_atoms);
    positions_1[0] = Vec3(-1, -2, -3);
    positions_1[1] = Vec3(0, 0, 0);
    positions_1[2] = Vec3(1, 0, 0);
    positions_1[3] = Vec3(0, 0, 2);
    positions_1[4] = Vec3(0, 0, 1);
    context.setPositions(positions_1);

    if (mode == "numerical_force") {
        double energy_1, energy_2, energy_3;

        vector<Vec3> forces;
        vector<Vec3> temp_positions;

        State state = context.getState(State::Forces | State::Energy | State::Positions);
        {
            forces = state.getForces();
            energy_1 = state.getPotentialEnergy();
            // printf("energy_1 = %lf\n", energy_1);
            temp_positions = state.getPositions();
            printf("forces:\n");
            for (int ii = 0; ii < num_of_atoms; ii ++) {
                print_Vec3(forces[ii]);
            }
            printf("energy = %f\n", energy_1);
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
                // printf("energy_2 = %lf\n", energy_2);
                numerical_derivatives[ii][jj] = (energy_2 - energy_1) / delta;
            }
        }
        // print out numerical results
        printf("numerical_derivatives = \n");
        for (int ii = 0; ii < num_of_atoms; ii ++) {
            print_Vec3(numerical_derivatives[ii]);
        }
    }
    else if (mode == "simulation") {
        for (int frameNum=1; ;++frameNum) {
            // Output current state information.
            OpenMM::State state    = context.getState(State::Positions | State::Forces);
            const double  timeInPs = state.getTime();
            writePdbFrame(frameNum, state); // output coordinates
            // printf("forces:\n");
            // auto forces = state.getForces();
            // for (int ii = 0; ii < num_of_atoms; ii ++) {
            //     print_Vec3(forces[ii]);
            // }
            if (timeInPs >= 100.)
                break;

            // Advance state many steps at a time, for efficient use of OpenMM.
            integrator.step(10); // (use a lot more than this normally)
        }    
    }
    return;
}

int main() 
{
    try {
        // cout << Platform::getDefaultPluginsDirectory();
        auto temp = OpenMM::Platform::loadPluginsFromDirectory("/home/fisiksnju/.anaconda2/lib/plugins");
        // auto temp = OpenMM::Platform::loadPluginsFromDirectory("/usr/local/openmm/lib/plugins");
        // for (auto item: temp) {
        //     cout << item;
        // }
        // registerExampleReferenceKernelFactories();
        test("numerical_force");
        // test_calculation_of_forces_by_comparing_with_numerical_derivatives_for_input_as_Cartesian_coordinates();
        return 0; // success!
    }
    // Catch and report usage and runtime errors detected by OpenMM and fail.
    catch(const std::exception& e) {
        printf("EXCEPTION: %s\n", e.what());
        return 1; // failure!
    }
}

// Handy homebrew PDB writer for quick-and-dirty trajectory output.
void writePdbFrame(int frameNum, const OpenMM::State& state) 
{
    // Reference atomic positions in the OpenMM State.
    const std::vector<OpenMM::Vec3>& posInNm = state.getPositions();

    // Use PDB MODEL cards to number trajectory frames
    printf("MODEL     %d\n", frameNum); // start of frame
    for (int a = 0; a < (int)posInNm.size(); ++a)
    {
        printf("ATOM  %5d  AR   AR     1    ", a+1); // atom number
        printf("%8.3f%8.3f%8.3f  1.00  0.00\n",      // coordinates
            // "*10" converts nanometers to Angstroms
            posInNm[a][0]*10, posInNm[a][1]*10, posInNm[a][2]*10);
    }
    printf("ENDMDL\n"); // end of frame
    return;
}
