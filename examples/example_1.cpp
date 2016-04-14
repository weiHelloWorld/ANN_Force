#include "OpenMM.h"
#include "../openmmapi/include/OpenMM_ANN.h"
#include "openmm/Platform.h"
#include "openmm/PluginInitializer.h"
#include "../../platforms/reference/ANN_ReferenceKernelFactory.cpp"

#include <cstdio>

using namespace OpenMM;

// Forward declaration of routine for printing one frame of the
// trajectory, defined later in this source file.
void writePdbFrame(int frameNum, const OpenMM::State&);

void simulateArgon()
{
    // Load any shared libraries containing GPU implementations.
    OpenMM::Platform::loadPluginsFromDirectory(
        OpenMM::Platform::getDefaultPluginsDirectory());

    // Create a system with nonbonded forces.
    OpenMM::System system;
    // OpenMM::NonbondedForce* nonbond = new OpenMM::NonbondedForce(); 
    // system.addForce(nonbond);
    int num_of_atoms = 6;
    for (int ii = 0; ii < num_of_atoms; ii ++) {
        system.addParticle(1.0);    
    }
    VerletIntegrator integrator(0.002);
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

    // Create three atoms.
    std::vector<OpenMM::Vec3> initPosInNm(num_of_atoms);
    for (int a = 0; a < num_of_atoms; ++a) 
    {
        system.addParticle(39.95); // mass of Ar, grams per mole
    }
    vector<Vec3> positions_1(num_of_atoms);
    positions_1[0] = Vec3(-1, -2, -3);
    positions_1[1] = Vec3(0, 0, 0);
    positions_1[2] = Vec3(1, 0, 0);
    positions_1[3] = Vec3(0, 0, 1);
    positions_1[4] = Vec3(0.5, 0, 0);
    positions_1[5] = Vec3(0, 0.3, 0.6);
    context.setPositions(positions_1);


    // Let OpenMM Context choose best platform.
    printf( "REMARK  Using OpenMM platform %s\n", 
        context.getPlatform().getName().c_str() );

    // Set starting positions of the atoms. Leave time and velocity zero.
    context.setPositions(initPosInNm);

    // Simulate.
    for (int frameNum=1; ;++frameNum) {
        // Output current state information.
        OpenMM::State state    = context.getState(OpenMM::State::Positions);
        const double  timeInPs = state.getTime();
        writePdbFrame(frameNum, state); // output coordinates

        if (timeInPs >= 20.)
            break;

        // Advance state many steps at a time, for efficient use of OpenMM.
        integrator.step(10); // (use a lot more than this normally)
    }
}

int main() 
{
    try {
        registerKernelFactories();  // this is required
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
}
