#include "ANN_ReferenceKernels.h"
#include "openmm/internal/ANN_ForceImpl.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "../../openmmapi/src/ANN_Force.cpp"

#include <cmath>
#include <iostream>
#include <stdio.h>


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
    int idx[4]; // indices of four atoms forming this dihedral
    int index_of_dihedral = index_of_cos_node_in_input_layer / 2;
    if (index_of_dihedral % 2 == 0) {
        idx[0] = index_of_backbone_atoms[3 * index_of_dihedral];
        idx[1] = index_of_backbone_atoms[3 * index_of_dihedral + 1];
        idx[2] = index_of_backbone_atoms[3 * index_of_dihedral + 2];
        idx[3] = index_of_backbone_atoms[3 * index_of_dihedral + 3];
    }
    else {
        idx[0] = index_of_backbone_atoms[3 * index_of_dihedral - 1];
        idx[1] = index_of_backbone_atoms[3 * index_of_dihedral];
        idx[2] = index_of_backbone_atoms[3 * index_of_dihedral + 1];
        idx[3] = index_of_backbone_atoms[3 * index_of_dihedral + 2];
    }

    RealVec diff_1 = positionData[idx[0]] - positionData[idx[1]];
    RealVec diff_2 = positionData[idx[1]] - positionData[idx[2]];
    RealVec diff_3 = positionData[idx[2]] - positionData[idx[3]];

    double x11 = diff_1[0], x12 = diff_1[1], x13 = diff_1[2];
    double x21 = diff_2[0], x22 = diff_2[1], x23 = diff_2[2]; 
    double x31 = diff_3[0], x32 = diff_3[1], x33 = diff_3[2]; 

    RealVec normal_1 = diff_1.cross(diff_2);
    RealVec normal_2 = diff_2.cross(diff_3);

    RealVec sin_vec = normal_1.cross(normal_2);
    int sign = (sin_vec[0] + sin_vec[1] + sin_vec[2]) * (diff_2[0] + diff_2[1] + diff_2[2]) > 0 ? 1 : -1;

    RealOpenMM n_1x = normal_1[0], n_1y = normal_1[1], n_1z = normal_1[2], 
               n_2x = normal_2[0], n_2y = normal_2[1], n_2z = normal_2[2];

    // following few lines are generated by sympy
    // note for der_of_cos_sin_to_nornal
    // first index: 0 cos, 1 sin
    // second index: components of normal_1 and normal_2
    RealOpenMM der_of_cos_sin_to_nornal[2][6]; 
    der_of_cos_sin_to_nornal[0][0] = sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(-n_1x*(n_1x*n_2x + n_1y*n_2y + n_1z*n_2z) + n_2x*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z));
    der_of_cos_sin_to_nornal[0][1] = sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(-n_1y*(n_1x*n_2x + n_1y*n_2y + n_1z*n_2z) + n_2y*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z));
    der_of_cos_sin_to_nornal[0][2] = sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(-n_1z*(n_1x*n_2x + n_1y*n_2y + n_1z*n_2z) + n_2z*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z));
    der_of_cos_sin_to_nornal[0][3] = sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(n_1x*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z) - n_2x*(n_1x*n_2x + n_1y*n_2y + n_1z*n_2z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z));
    der_of_cos_sin_to_nornal[0][4] = sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(n_1y*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z) - n_2y*(n_1x*n_2x + n_1y*n_2y + n_1z*n_2z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z));
    der_of_cos_sin_to_nornal[0][5] = sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(n_1z*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z) - n_2z*(n_1x*n_2x + n_1y*n_2y + n_1z*n_2z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z));
    der_of_cos_sin_to_nornal[1][0] = sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(-n_1x*((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y)) + (n_2y*(n_1x*n_2y - n_1y*n_2x) + n_2z*(n_1x*n_2z - n_1z*n_2x))*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*sqrt((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y))) * sign;
    der_of_cos_sin_to_nornal[1][1] = sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(-n_1y*((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y)) + (-n_2x*(n_1x*n_2y - n_1y*n_2x) + n_2z*(n_1y*n_2z - n_1z*n_2y))*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*sqrt((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y))) * sign;
    der_of_cos_sin_to_nornal[1][2] = -sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(n_1z*((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y)) + (n_2x*(n_1x*n_2z - n_1z*n_2x) + n_2y*(n_1y*n_2z - n_1z*n_2y))*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*sqrt((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y))) * sign;
    der_of_cos_sin_to_nornal[1][3] = -sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(n_2x*((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y)) + (n_1y*(n_1x*n_2y - n_1y*n_2x) + n_1z*(n_1x*n_2z - n_1z*n_2x))*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*sqrt((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y))) * sign;
    der_of_cos_sin_to_nornal[1][4] = sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(-n_2y*((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y)) + (n_1x*(n_1x*n_2y - n_1y*n_2x) - n_1z*(n_1y*n_2z - n_1z*n_2y))*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*sqrt((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y))) * sign;
    der_of_cos_sin_to_nornal[1][5] = sqrt((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))*(-n_2z*((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y)) + (n_1x*(n_1x*n_2z - n_1z*n_2x) + n_1y*(n_1y*n_2z - n_1z*n_2y))*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z))/((n_1x*n_1x + n_1y*n_1y + n_1z*n_1z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*(n_2x*n_2x + n_2y*n_2y + n_2z*n_2z)*sqrt((n_1x*n_2y - n_1y*n_2x)*(n_1x*n_2y - n_1y*n_2x) + (n_1x*n_2z - n_1z*n_2x)*(n_1x*n_2z - n_1z*n_2x) + (n_1y*n_2z - n_1z*n_2y)*(n_1y*n_2z - n_1z*n_2y))) * sign;

    // calculate derivatives of normal with respect to diff vectors
    RealOpenMM der_of_normal_to_diff[6][9];
    der_of_normal_to_diff[0][0] = 0;
    der_of_normal_to_diff[0][1] = x23;
    der_of_normal_to_diff[0][2] = -x22;
    der_of_normal_to_diff[0][3] = 0;
    der_of_normal_to_diff[0][4] = -x13;
    der_of_normal_to_diff[0][5] = x12;
    der_of_normal_to_diff[0][6] = 0;
    der_of_normal_to_diff[0][7] = 0;
    der_of_normal_to_diff[0][8] = 0;
    der_of_normal_to_diff[1][0] = -x23;
    der_of_normal_to_diff[1][1] = 0;
    der_of_normal_to_diff[1][2] = x21;
    der_of_normal_to_diff[1][3] = x13;
    der_of_normal_to_diff[1][4] = 0;
    der_of_normal_to_diff[1][5] = -x11;
    der_of_normal_to_diff[1][6] = 0;
    der_of_normal_to_diff[1][7] = 0;
    der_of_normal_to_diff[1][8] = 0;
    der_of_normal_to_diff[2][0] = x22;
    der_of_normal_to_diff[2][1] = -x21;
    der_of_normal_to_diff[2][2] = 0;
    der_of_normal_to_diff[2][3] = -x12;
    der_of_normal_to_diff[2][4] = x11;
    der_of_normal_to_diff[2][5] = 0;
    der_of_normal_to_diff[2][6] = 0;
    der_of_normal_to_diff[2][7] = 0;
    der_of_normal_to_diff[2][8] = 0;
    der_of_normal_to_diff[3][0] = 0;
    der_of_normal_to_diff[3][1] = 0;
    der_of_normal_to_diff[3][2] = 0;
    der_of_normal_to_diff[3][3] = 0;
    der_of_normal_to_diff[3][4] = x33;
    der_of_normal_to_diff[3][5] = -x32;
    der_of_normal_to_diff[3][6] = 0;
    der_of_normal_to_diff[3][7] = -x23;
    der_of_normal_to_diff[3][8] = x22;
    der_of_normal_to_diff[4][0] = 0;
    der_of_normal_to_diff[4][1] = 0;
    der_of_normal_to_diff[4][2] = 0;
    der_of_normal_to_diff[4][3] = -x33;
    der_of_normal_to_diff[4][4] = 0;
    der_of_normal_to_diff[4][5] = x31;
    der_of_normal_to_diff[4][6] = x23;
    der_of_normal_to_diff[4][7] = 0;
    der_of_normal_to_diff[4][8] = -x21;
    der_of_normal_to_diff[5][0] = 0;
    der_of_normal_to_diff[5][1] = 0;
    der_of_normal_to_diff[5][2] = 0;
    der_of_normal_to_diff[5][3] = x32;
    der_of_normal_to_diff[5][4] = -x31;
    der_of_normal_to_diff[5][5] = 0;
    der_of_normal_to_diff[5][6] = -x22;
    der_of_normal_to_diff[5][7] = x21;
    der_of_normal_to_diff[5][8] = 0;

    // chain rule, combine previous two results
    RealOpenMM der_of_cos_sin_to_diff[2][9];
    for(int ii = 0; ii < 2; ii ++) {
        for (int jj = 0; jj < 9; jj ++) {
            der_of_cos_sin_to_diff[ii][jj] = 0;
            for (int kk = 0; kk < 6; kk ++) {
                der_of_cos_sin_to_diff[ii][jj] += der_of_cos_sin_to_nornal[ii][kk] * der_of_normal_to_diff[kk][jj];
            }
        }
    }
    for (int ii = 0; ii < 3; ii ++) {
        for (int jj = 0; jj < 3 ; jj ++) {
            auto temp = + derivatives_of_first_layer[index_of_cos_node_in_input_layer] 
                                        * der_of_cos_sin_to_diff[0][3 * ii + jj]
                                        + derivatives_of_first_layer[index_of_sin_node_in_input_layer] 
                                        * der_of_cos_sin_to_diff[1][3 * ii + jj]; 
            forceData[idx[ii]][jj] += + temp;
            forceData[idx[ii + 1]][jj] += -temp;
#ifdef DEBUG
            printf("temp = %f\n", temp);
            printf("forceData[%d][%d] = %f\n", idx[ii], jj, forceData[idx[ii]][jj]);
            printf("forceData[%d][%d] = %f\n", idx[ii + 1], jj, forceData[idx[ii + 1]][jj]);
#endif
        }
    }

#ifdef DEBUG
    printf("der_of_cos_sin_to_diff = \n");
    int num_of_rows = 2;
    int num_of_cols = 9;
    for (int ii = 0; ii < num_of_rows; ii ++) {
        for (int jj = 0; jj < num_of_cols; jj ++) {
            printf("%lf\t", der_of_cos_sin_to_diff[ii][jj]);
        }
        printf("\n");
    }
#endif

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