#include "ANN_ReferenceKernels.h"
#include "openmm/internal/ANN_ForceImpl.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "../../openmmapi/src/ANN_Force.cpp"

#include <cmath>
#include <iostream>
#include <stdio.h>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->velocities);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

static RealVec& extractBoxSize(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *(RealVec*) data->periodicBoxSize;
}

static RealVec* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (RealVec*) data->periodicBoxVectors;
}

// ***************************************************************************

ReferenceCalcANN_ForceKernel::ReferenceCalcANN_ForceKernel(std::string name, const Platform& platform, const System& system) : 
                CalcANN_ForceKernel(name, platform), system(system) {
}

ReferenceCalcANN_ForceKernel::~ReferenceCalcANN_ForceKernel() {
    // TODO: delete the coefficients allocated
}

void ReferenceCalcANN_ForceKernel::initialize(const System& system, const ANN_Force& force) {
    num_of_nodes = force.get_num_of_nodes();
    index_of_backbone_atoms = force.get_index_of_backbone_atoms();
    layer_types = force.get_layer_types();
    values_of_biased_nodes = force.get_values_of_biased_nodes();
    potential_center = force.get_potential_center();
    force_constant = force.get_force_constant();
    assert (potential_center.size() == num_of_nodes[NUM_OF_LAYERS - 1]);
    // now deal with coefficients of connections
    auto temp_coeff = force.get_coeffients_of_connections(); // FIXME: modify this initialization later
    for (int ii = 0; ii < NUM_OF_LAYERS - 1; ii ++) {
        int num_of_rows, num_of_cols; // num of rows/cols for the coeff matrix of this connection
        num_of_rows = num_of_nodes[ii + 1];
        num_of_cols = num_of_nodes[ii];
        assert (num_of_rows * num_of_cols == temp_coeff[ii].size()); // check whether the size matches
        // create a 2d array to hold coefficients begin
        coeff[ii] = new double*[num_of_rows];
        for (int kk = 0; kk < num_of_rows; kk ++) {
            coeff[ii][kk] = new double[num_of_cols];
        }
        // creation end
        for (int jj = 0; jj < temp_coeff[ii].size(); jj ++) {
            coeff[ii][jj / num_of_cols][jj % num_of_cols] = temp_coeff[ii][jj];
        }
    }
    return;
}

double ReferenceCalcANN_ForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& posData   = extractPositions(context);
    vector<RealVec>& forceData = extractForces(context);
    RealOpenMM energy      = calculateForceAndEnergy(posData, forceData); 
                                // the output of force for each atom is stored in forceData
    return static_cast<double>(energy);
}

void ReferenceCalcANN_ForceKernel::copyParametersToContext(ContextImpl& context, const ANN_Force& force) {
    // if (numBonds != force.getNumBonds())
    //     throw OpenMMException("updateParametersInContext: The number of bonds has changed");

    // // Record the values.

    // for (int i = 0; i < numBonds; ++i) {
    //     int particle1Index, particle2Index;
    //     double lengthValue, kValue;
    //     force.getBondParameters(i, particle1Index, particle2Index, lengthValue, kValue);
    //     if (particle1Index != particle1[i] || particle2Index != particle2[i])
    //         throw OpenMMException("updateParametersInContext: The set of particles in a bond has changed");
    //     length[i] = (RealOpenMM) lengthValue;
    //     kQuadratic[i] = (RealOpenMM) kValue;
    // }
}

RealOpenMM ReferenceCalcANN_ForceKernel::calculateForceAndEnergy(vector<RealVec>& positionData, vector<RealVec>& forceData) {
    // test case: add force on first atom, fix it at (0,0,0)
    RealOpenMM coef = 100.0;
    forceData[0][0]    += - coef * (positionData[0][0] - 0.1);
    forceData[0][1]    += - coef * (positionData[0][1] - 0.2);
    forceData[0][2]    += - coef * positionData[0][2];
    return 0;  // TODO: fix this later
}


void ReferenceCalcANN_ForceKernel::calculate_output_of_each_layer(const vector<RealOpenMM>& input) {
    // first layer
    output_of_each_layer[0] = input;
    // following layers
    for(int ii = 1; ii < NUM_OF_LAYERS; ii ++) {
        vector<double> temp_input_of_this_layer = vector<double>(num_of_nodes[ii]);
        output_of_each_layer[ii].resize(num_of_nodes[ii]);
        // first calculate input
        for (int jj = 0; jj < num_of_nodes[ii]; jj ++) {
            temp_input_of_this_layer[jj] = values_of_biased_nodes[ii - 1][jj];  // add bias term
            for (int kk = 0; kk < num_of_nodes[ii - 1]; kk ++) {
                temp_input_of_this_layer[jj] += coeff[ii - 1][jj][kk] * output_of_each_layer[ii - 1][kk];
            }
        }
        // then get output
        if (layer_types[ii - 1] == string("Linear")) {
            for(int jj = 0; jj < num_of_nodes[ii]; jj ++) {
                output_of_each_layer[ii][jj] = temp_input_of_this_layer[jj];
            }
        }
        else if (layer_types[ii - 1] == string("Tanh")) {
            for(int jj = 0; jj < num_of_nodes[ii]; jj ++) {
                output_of_each_layer[ii][jj] = tanh(temp_input_of_this_layer[jj]);
            }
        }
    }
    return;
}

void ReferenceCalcANN_ForceKernel::back_prop(vector<double>& derivatives_for_input) {
    // the return value is the derivatives with respect to the input in the first layer
    // return value is stored in "derivatives_for_input"
    auto temp_derivatives_of_each_layer = output_of_each_layer;  // the data structure and size should be the same, so I simply deep copy it
    // first calculate derivatives for bottleneck layer
    for (int ii = 0; ii < num_of_nodes[NUM_OF_LAYERS - 1]; ii ++) {
        temp_derivatives_of_each_layer[NUM_OF_LAYERS - 1][ii] = output_of_each_layer[NUM_OF_LAYERS - 1][ii] \
                                                                - potential_center[ii];
    }
    // the use back propagation to calculate derivatives for previous layers
    for (int jj = NUM_OF_LAYERS - 2; jj >= 0; jj --) {
        for (int mm = 0; mm = num_of_nodes[jj]; mm ++) {
            temp_derivatives_of_each_layer[jj][mm] = 0;
            for (int kk = 0; kk = num_of_nodes[jj + 1]; kk ++) {
                temp_derivatives_of_each_layer[jj][mm] += temp_derivatives_of_each_layer[jj + 1][kk] \
                                * coeff[jj][kk][mm] \
                                * (1 - output_of_each_layer[jj + 1][kk] * output_of_each_layer[jj + 1][kk]);
                                // TODO: here we assume that it is Tanh Layer, fix this later.
            }
        }
    }
    derivatives_for_input = temp_derivatives_of_each_layer[0];
    return;
}

void ReferenceCalcANN_ForceKernel::get_cos_and_sin_of_dihedral_angles(const vector<RealVec>& positionData,
                                                                            vector<RealOpenMM>& cos_sin_value) {
    assert (index_of_backbone_atoms.size() % 3 == 0);
    RealOpenMM temp_cos, temp_sin;
    for (int ii = 0; ii < index_of_backbone_atoms.size() / 3; ii ++) {
        if (ii != 0) {
            get_cos_and_sin_for_four_atoms(index_of_backbone_atoms[3 * ii - 1], index_of_backbone_atoms[3 * ii], 
                                            index_of_backbone_atoms[3 * ii + 1], index_of_backbone_atoms[3 * ii + 2], 
                                            positionData, temp_cos, temp_sin);
            cos_sin_value.push_back(temp_cos);
            cos_sin_value.push_back(temp_sin);
        }
        if (ii != index_of_backbone_atoms.size() / 3 - 1) {
            get_cos_and_sin_for_four_atoms(index_of_backbone_atoms[3 * ii], index_of_backbone_atoms[3 * ii + 1], 
                                            index_of_backbone_atoms[3 * ii + 2], index_of_backbone_atoms[3 * ii + 3], 
                                            positionData, temp_cos, temp_sin);
            cos_sin_value.push_back(temp_cos);
            cos_sin_value.push_back(temp_sin);
        }
    }
#ifdef DEBUG
    assert (cos_sin_value.size() == index_of_backbone_atoms.size() / 3 * 2);
    assert (cos_sin_value.size() == num_of_nodes[0]);
#endif
    return;
}

void ReferenceCalcANN_ForceKernel::get_cos_and_sin_for_four_atoms(int idx_1, int idx_2, int idx_3, int idx_4, 
                                const vector<RealVec>& positionData, RealOpenMM& cos_value, RealOpenMM& sin_value) {
    RealVec diff_1 = positionData[idx_1] - positionData[idx_2];
    RealVec diff_2 = positionData[idx_2] - positionData[idx_3];
    RealVec diff_3 = positionData[idx_3] - positionData[idx_4];


    RealVec normal_1 = diff_1.cross(diff_2);
    RealVec normal_2 = diff_2.cross(diff_3);
    normal_1 /= sqrt(normal_1.dot(normal_1));  // normalization
    normal_2 /= sqrt(normal_2.dot(normal_2));
    cos_value = normal_1.dot(normal_2);
    RealVec sin_vec = normal_1.cross(normal_2);
    int sign = (sin_vec[0] + sin_vec[1] + sin_vec[2]) * (diff_2[0] + diff_2[1] + diff_2[2]) > 0 ? 1 : -1;
    sin_value = sqrt(sin_vec.dot(sin_vec)) * sign;
#ifdef DEBUG    
    // printf("%f,%f,%f\n", diff_1[0], diff_1[1], diff_1[2]);
    // printf("%f,%f,%f\n", diff_2[0], diff_2[1], diff_2[2]);
    // printf("%f,%f,%f\n", diff_3[0], diff_3[1], diff_3[2]);
    // printf("%f,%f,%f\n", normal_1[0], normal_1[1], normal_1[2]);
    // printf("%f,%f,%f\n", normal_2[0], normal_2[1], normal_2[2]);
    // printf("%f,%f,%f\n", sin_vec[0], sin_vec[1], sin_vec[2]);
    // printf("%f\n", sin_value);
#endif
#ifdef DEBUG
    assert (abs(cos_value * cos_value + sin_value * sin_value - 1 ) < 1e-5);
#endif
    return;
}