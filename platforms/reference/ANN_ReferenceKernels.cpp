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

RealOpenMM ReferenceCalcANN_ForceKernel::candidate_1(vector<RealVec>& positionData, vector<RealVec>& forceData) {
    // test case: add force on first atom, fix it at (0,0,0)
    RealOpenMM coef = 100.0;
    forceData[0][0]    += - coef * (positionData[0][0] - 0.1);
    forceData[0][1]    += - coef * (positionData[0][1] - 0.2);
    forceData[0][2]    += - coef * positionData[0][2];
    return 0;  // TODO: fix this later
}

RealOpenMM ReferenceCalcANN_ForceKernel::candidate_2(vector<RealVec>& positionData, vector<RealVec>& forceData) {
    // test case
    vector<RealOpenMM> cos_sin_value;
    get_cos_and_sin_of_dihedral_angles(positionData, cos_sin_value);
    calculate_output_of_each_layer(cos_sin_value);
    vector<vector<double> > derivatives_of_each_layer;
    back_prop(derivatives_of_each_layer);
    get_force_from_derivative_of_first_layer(0, 1, positionData, forceData, derivatives_of_each_layer[0]); // TODO: here we only include the first dihedral, add others later
    return 0;  // TODO: fix this later
}

RealOpenMM ReferenceCalcANN_ForceKernel::calculateForceAndEnergy(vector<RealVec>& positionData, vector<RealVec>& forceData) {
    return candidate_2(positionData, forceData);
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
#ifdef DEBUG
    // print out the result for debugging
    printf("output_of_each_layer = \n");
    for (int ii = 0; ii < NUM_OF_LAYERS; ii ++) {
        printf("layer[%d]: ", ii);
        if (ii != 0) {
            cout << layer_types[ii - 1] << "\t";    
        }
        else {
            cout << "input \t" ;
        }
        for (int jj = 0; jj < num_of_nodes[ii]; jj ++) {
            printf("%lf\t", output_of_each_layer[ii][jj]);
        }
        printf("\n");
    }
    printf("\n");
#endif
    return;
}

void ReferenceCalcANN_ForceKernel::back_prop(vector<vector<double> >& derivatives_of_each_layer) {
    derivatives_of_each_layer = output_of_each_layer;  // the data structure and size should be the same, so I simply deep copy it
    // first calculate derivatives for bottleneck layer
    for (int ii = 0; ii < num_of_nodes[NUM_OF_LAYERS - 1]; ii ++) {
        derivatives_of_each_layer[NUM_OF_LAYERS - 1][ii] = (output_of_each_layer[NUM_OF_LAYERS - 1][ii] \
                                                                - potential_center[ii]) * force_constant;
    }
    // the use back propagation to calculate derivatives for previous layers
    for (int jj = NUM_OF_LAYERS - 2; jj >= 0; jj --) {
        for (int mm = 0; mm < num_of_nodes[jj]; mm ++) {
            derivatives_of_each_layer[jj][mm] = 0;
            for (int kk = 0; kk < num_of_nodes[jj + 1]; kk ++) {
                if (layer_types[jj] == string("Tanh")) {
                    // printf("tanh\n");
                    derivatives_of_each_layer[jj][mm] += derivatives_of_each_layer[jj + 1][kk] \
                                * coeff[jj][kk][mm] \
                                * (1 - output_of_each_layer[jj + 1][kk] * output_of_each_layer[jj + 1][kk]);
#ifdef DEBUG
                    // printf("this:\n");
                    // printf("%lf\n", derivatives_of_each_layer[jj + 1][kk]);
                    // printf("%lf\n", coeff[jj][kk][mm]);
                    // printf("%lf\n", (1 - output_of_each_layer[jj + 1][kk] * output_of_each_layer[jj + 1][kk]));
#endif
                }
                else if (layer_types[jj] == string("Linear")) {
                    // printf("linear\n");
                    derivatives_of_each_layer[jj][mm] += derivatives_of_each_layer[jj + 1][kk] \
                                * coeff[jj][kk][mm] \
                                * 1;
                }
            }
        }
    }
#ifdef DEBUG
    // print out the result for debugging
    printf("derivatives_of_each_layer = \n");
    for (int ii = 0; ii < NUM_OF_LAYERS; ii ++) {
        printf("layer[%d]: ", ii);
        for (int jj = 0; jj < num_of_nodes[ii]; jj ++) {
            printf("%lf\t", derivatives_of_each_layer[ii][jj]);
        }
        printf("\n");
    }
    printf("\n");
#endif
    return;
}

void ReferenceCalcANN_ForceKernel::get_force_from_derivative_of_first_layer(int index_of_cos_node_in_input_layer, 
                                                                            int index_of_sin_node_in_input_layer,
                                                                            vector<RealVec>& positionData,
                                                                            vector<RealVec>& forceData,
                                                                            vector<double>& derivatives_of_first_layer) {
    /**
     * this function calculates force (the derivative of potential with respect to the Cartesian coordinates for four atoms),
     * from the derivatives with respect to the inputs in the first layer
     * INPUT: two indices of nodes in the input layer, corresponding to cos_value and sin_value respectively
     * NO OUTPUT, it updates the 'forceData' for these four atoms
     */
    // first work with cos value
    int idx_1, idx_2, idx_3, idx_4; // indices of four atoms forming this dihedral
    int index_of_dihedral = index_of_cos_node_in_input_layer / 2;
    if (index_of_dihedral % 2 == 0) {
        idx_1 = index_of_backbone_atoms[3 * index_of_dihedral];
        idx_2 = index_of_backbone_atoms[3 * index_of_dihedral + 1];
        idx_3 = index_of_backbone_atoms[3 * index_of_dihedral + 2];
        idx_4 = index_of_backbone_atoms[3 * index_of_dihedral + 3];
    }
    else {
        idx_1 = index_of_backbone_atoms[3 * index_of_dihedral - 1];
        idx_2 = index_of_backbone_atoms[3 * index_of_dihedral];
        idx_3 = index_of_backbone_atoms[3 * index_of_dihedral + 1];
        idx_4 = index_of_backbone_atoms[3 * index_of_dihedral + 2];
    }

    RealVec diff_1 = positionData[idx_1] - positionData[idx_2];
    RealVec diff_2 = positionData[idx_2] - positionData[idx_3];
    RealVec diff_3 = positionData[idx_3] - positionData[idx_4];

    RealVec normal_1 = diff_1.cross(diff_2);
    RealVec normal_2 = diff_2.cross(diff_3);

    double v1_squared = normal_1.dot(normal_1);
    double v2_squared = normal_2.dot(normal_2);
    double v1_x = normal_1[0], v1_y = normal_1[1], v1_z = normal_1[2]; 
    double v2_x = normal_2[0], v2_y = normal_2[1], v2_z = normal_2[2];
    double x11 = diff_1[0], x12 = diff_1[1], x13 = diff_1[2];
    double x21 = diff_2[0], x22 = diff_2[1], x23 = diff_2[2]; 
    double x31 = diff_3[0], x32 = diff_3[1], x33 = diff_3[2]; 
    // first deal with derivatives related to diff_1
    double der_of_cos_to_diff_1_x = (v1_squared*(-v2_y*x23 + v2_z*x22)
                                - (-v1_y*x23 + v1_z*x22)*(v1_x*v2_x + v1_y*v2_y + v1_z*v2_z))
                                /(v1_squared * sqrt(v1_squared) * sqrt(v2_squared)); // this is derivative of cos value with respect to x component of diff_1
    double der_of_cos_to_diff_1_y = (v1_squared*(v2_x*x23 - v2_z*x21) + (-v1_x*x23 + v1_z*x21)*(v1_x*v2_x + v1_y*v2_y + v1_z*v2_z))/(v1_squared * sqrt(v1_squared) * sqrt(v2_squared));
    double der_of_cos_to_diff_1_z = (-v1_squared*(v2_x*x22 - v2_y*x21) + (v1_x*x22 - v1_y*x21)*(v1_x*v2_x + v1_y*v2_y + v1_z*v2_z))/(v1_squared * sqrt(v1_squared) * sqrt(v2_squared));
    forceData[idx_1][0] += + der_of_cos_to_diff_1_x * derivatives_of_first_layer[index_of_cos_node_in_input_layer];
    forceData[idx_2][0] += - der_of_cos_to_diff_1_x * derivatives_of_first_layer[index_of_cos_node_in_input_layer];
    forceData[idx_1][1] += + der_of_cos_to_diff_1_y * derivatives_of_first_layer[index_of_cos_node_in_input_layer];
    forceData[idx_2][1] += - der_of_cos_to_diff_1_y * derivatives_of_first_layer[index_of_cos_node_in_input_layer];
    forceData[idx_1][2] += + der_of_cos_to_diff_1_z * derivatives_of_first_layer[index_of_cos_node_in_input_layer];
    forceData[idx_2][2] += - der_of_cos_to_diff_1_z * derivatives_of_first_layer[index_of_cos_node_in_input_layer];
    // now diff_2
    // now diff_3
    // TODO

    return;
}

void ReferenceCalcANN_ForceKernel::get_cos_and_sin_of_dihedral_angles(const vector<RealVec>& positionData,
                                                                            vector<RealOpenMM>& cos_sin_value) {
    assert (index_of_backbone_atoms.size() % 3 == 0);
#ifdef DEBUG
    printf("size of index_of_backbone_atoms = %lu\n", index_of_backbone_atoms.size());
#endif
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
    printf("cos_sin_value.size() = %d, num_of_nodes[0] = %d\n", cos_sin_value.size(), num_of_nodes[0]);
    assert (cos_sin_value.size() == index_of_backbone_atoms.size() / 3 * 4 - 4); // FIXME: correct?
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
    printf("idx_1 = %d, idx_2 = %d, idx_3 = %d, idx_4 = %d\n", idx_1, idx_2, idx_3, idx_4);
    printf("cos_value = %lf, sin_value = %lf\n", cos_value, sin_value);
    assert (abs(cos_value * cos_value + sin_value * sin_value - 1 ) < 1e-5);
#endif
    return;
}