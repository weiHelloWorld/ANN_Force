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

void simulateArgon()
{
    // Load any shared libraries containing GPU implementations.
    // OpenMM::Platform::loadPluginsFromDirectory(
    //     OpenMM::Platform::getDefaultPluginsDirectory());

    // Create a system with nonbonded forces.
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
    cout << "data_type_in_input_layer = " << forceField -> get_data_type_in_input_layer() << endl;
    system.addForce(forceField);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integrator, platform);
    vector<Vec3> positions_1(num_of_atoms);
    positions_1[0] = Vec3(-1, -2, -3);
    positions_1[1] = Vec3(0, 0, 0);
    positions_1[2] = Vec3(1, 0, 0);
    positions_1[3] = Vec3(0, 0, 2);
    context.setPositions(positions_1);


    // Simulate.
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
        if (timeInPs >= 200.)
            break;

        // Advance state many steps at a time, for efficient use of OpenMM.
        integrator.step(10); // (use a lot more than this normally)
    }
    return;
}

int main() 
{
    try {
        // cout << Platform::getDefaultPluginsDirectory();
        auto temp = OpenMM::Platform::loadPluginsFromDirectory(
                "/home/fisiksnju/.anaconda2/lib/plugins");
        // for (auto item: temp) {
        //     cout << item;
        // }
        // registerExampleReferenceKernelFactories();
        simulateArgon();
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
