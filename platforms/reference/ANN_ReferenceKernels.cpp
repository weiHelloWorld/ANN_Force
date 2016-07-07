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
    list_of_index_of_atoms_forming_dihedrals = force.get_list_of_index_of_atoms_forming_dihedrals();
    layer_types = force.get_layer_types();
    values_of_biased_nodes = force.get_values_of_biased_nodes();
    potential_center = force.get_potential_center();
    force_constant = force.get_force_constant();
    if (layer_types[NUM_OF_LAYERS - 2] != string("Circular")) {
        assert (potential_center.size() == num_of_nodes[NUM_OF_LAYERS - 1]);
    }
    else {
        assert (potential_center.size() * 2 == num_of_nodes[NUM_OF_LAYERS - 1]);
    }
    
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
    return 0;  
}

RealOpenMM ReferenceCalcANN_ForceKernel::candidate_2(vector<RealVec>& positionData, vector<RealVec>& forceData) {
    // test case
    vector<RealOpenMM> cos_sin_value;
    get_cos_and_sin_of_dihedral_angles(positionData, cos_sin_value);
#ifdef DEBUG
    auto index_of_backbone = std::vector<int> {2,5,7,9,15,17,19};
    for (auto idx: index_of_backbone) {
        printf("%f, %f, %f, ", positionData[idx-1][0], positionData[idx-1][1], positionData[idx-1][2]);
    }
    printf("\n");
#endif
    calculate_output_of_each_layer(cos_sin_value);
    vector<vector<double> > derivatives_of_each_layer;
    back_prop(derivatives_of_each_layer);
    get_all_forces_from_derivative_of_first_layer(positionData, forceData, derivatives_of_each_layer[0]);
    // for (auto my_f: forceData) {
    //     printf("force = %f\t%f\t%f\n", my_f[0], my_f[1], my_f[2]);
    // }
    // printf("\n");
    return update_and_get_potential_energy();  
}

RealOpenMM ReferenceCalcANN_ForceKernel::calculateForceAndEnergy(vector<RealVec>& positionData, vector<RealVec>& forceData) {
    return candidate_2(positionData, forceData);
}



void ReferenceCalcANN_ForceKernel::calculate_output_of_each_layer(const vector<RealOpenMM>& input) {
    // first layer
    output_of_each_layer[0] = input;
    // following layers
    for(int ii = 1; ii < NUM_OF_LAYERS; ii ++) {
        output_of_each_layer[ii].resize(num_of_nodes[ii]);
        input_of_each_layer[ii].resize(num_of_nodes[ii]);
        // first calculate input
        for (int jj = 0; jj < num_of_nodes[ii]; jj ++) {
            input_of_each_layer[ii][jj] = values_of_biased_nodes[ii - 1][jj];  // add bias term
            for (int kk = 0; kk < num_of_nodes[ii - 1]; kk ++) {
                input_of_each_layer[ii][jj] += coeff[ii - 1][jj][kk] * output_of_each_layer[ii - 1][kk];
            }
        }
        // then get output
        if (layer_types[ii - 1] == string("Linear")) {
            for(int jj = 0; jj < num_of_nodes[ii]; jj ++) {
                output_of_each_layer[ii][jj] = input_of_each_layer[ii][jj];
            }
        }
        else if (layer_types[ii - 1] == string("Tanh")) {
            for(int jj = 0; jj < num_of_nodes[ii]; jj ++) {
                output_of_each_layer[ii][jj] = tanh(input_of_each_layer[ii][jj]);
            }
        }
        else if (layer_types[ii - 1] == string("Circular")) {
            assert (num_of_nodes[ii] % 2 == 0);
            for(int jj = 0; jj < num_of_nodes[ii] / 2; jj ++) {
                double radius = sqrt(input_of_each_layer[ii][2 * jj] * input_of_each_layer[ii][2 * jj] 
                                    +input_of_each_layer[ii][2 * jj + 1] * input_of_each_layer[ii][2 * jj + 1]);
                output_of_each_layer[ii][2 * jj] = input_of_each_layer[ii][2 * jj] / radius;
                output_of_each_layer[ii][2 * jj + 1] = input_of_each_layer[ii][2 * jj + 1] / radius;
#ifdef DEBUG
                if (abs(output_of_each_layer[ii][2 * jj] * output_of_each_layer[ii][2 * jj]
                      + output_of_each_layer[ii][2 * jj + 1] * output_of_each_layer[ii][2 * jj + 1] - 1) > 1e-5) {
                    printf("error: two values are %lf, %lf\n", output_of_each_layer[ii][2 * jj], 
                                                               output_of_each_layer[ii][2 * jj + 1]);
                }
#endif
            }
        }
        else {
            printf("layer type not found!\n\n");
            return;
        }
    }
#ifdef DEBUG
    // print out the result for debugging
    printf("output_of_each_layer = \n");
    // for (int ii = NUM_OF_LAYERS - 1; ii < NUM_OF_LAYERS; ii ++) {
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
    if (layer_types[NUM_OF_LAYERS - 2] != string("Circular")) {
        for (int ii = 0; ii < num_of_nodes[NUM_OF_LAYERS - 1]; ii ++) {
            derivatives_of_each_layer[NUM_OF_LAYERS - 1][ii] = (output_of_each_layer[NUM_OF_LAYERS - 1][ii] \
                                                                    - potential_center[ii]) * force_constant;
        }
    }
    else {
        // FIXME: fix for circular layer
        for (int ii = 0; ii < num_of_nodes[NUM_OF_LAYERS - 1] / 2; ii ++) {
#ifdef DEBUG
            assert(output_of_each_layer[NUM_OF_LAYERS - 1].size() == 2 * potential_center.size());
#endif
            double cos_value = output_of_each_layer[NUM_OF_LAYERS - 1][2 * ii];
            double sin_value = output_of_each_layer[NUM_OF_LAYERS - 1][2 * ii + 1];
            int sign = sin_value > 0 ? 1 : -1;
            double angle = acos(cos_value) * sign;
            double angle_distance_1 = angle - potential_center[ii];
            double angle_distance_2 = angle_distance_1 + 6.2832;
            double angle_distance_3 = angle_distance_1 - 6.2832;
            // need to take periodicity into account
            double temp_distance = angle_distance_1;
            if (abs(angle_distance_2) < abs(temp_distance) ) {
                temp_distance = angle_distance_2;
            }
            if (abs(angle_distance_3) < abs(temp_distance) ) {
                temp_distance = angle_distance_3;
            }
            derivatives_of_each_layer[NUM_OF_LAYERS - 1][2 * ii] = force_constant * temp_distance * (-1)\
                                                         / sqrt(1 - cos_value * cos_value) * sign;
            derivatives_of_each_layer[NUM_OF_LAYERS - 1][2 * ii + 1] = 0;  // FIXME: may I set it to be simply 0?
        }
    }

    // the use back propagation to calculate derivatives for previous layers
    for (int jj = NUM_OF_LAYERS - 2; jj >= 0; jj --) {
        if (layer_types[jj] == string("Circular")) {
            vector<double> temp_derivative_of_input_for_this_layer;
            temp_derivative_of_input_for_this_layer.resize(num_of_nodes[jj + 1]);
#ifdef DEBUG
            assert (num_of_nodes[jj + 1] % 2 == 0);
#endif
            // first calculate the derivative of input from derivative of output of this circular layer
            for(int ii = 0; ii < num_of_nodes[jj + 1] / 2; ii ++) {
                // printf("size of input_of_each_layer[%d] = %d\n",jj,  input_of_each_layer[jj].size());
                double x_p = input_of_each_layer[jj + 1][2 * ii];
                double x_q = input_of_each_layer[jj + 1][2 * ii + 1];
                double radius = sqrt(x_p * x_p + x_q * x_q);
                temp_derivative_of_input_for_this_layer[2 * ii] = x_q / (radius * radius * radius) 
                                                        * (x_q * derivatives_of_each_layer[jj + 1][2 * ii] 
                                                        - x_p * derivatives_of_each_layer[jj + 1][2 * ii + 1]);
                temp_derivative_of_input_for_this_layer[2 * ii + 1] = x_p / (radius * radius * radius) 
                                                        * (x_p * derivatives_of_each_layer[jj + 1][2 * ii + 1] 
                                                        - x_q * derivatives_of_each_layer[jj + 1][2 * ii]);
            }
#ifdef DEBUG
            for (int mm = 0; mm < num_of_nodes[jj + 1]; mm ++) {
                // printf("temp_derivative_of_input_for_this_layer[%d] = %lf\n", mm, temp_derivative_of_input_for_this_layer[mm]);
                // printf("derivatives_of_each_layer[%d + 1][%d] = %lf\n", jj, mm, derivatives_of_each_layer[jj + 1][mm]);
            }
#endif
            // the calculate the derivative of output of layer jj, from derivative of input of layer (jj + 1)
            for (int mm = 0; mm < num_of_nodes[jj]; mm ++) {
                derivatives_of_each_layer[jj][mm] = 0;
                for (int kk = 0; kk < num_of_nodes[jj + 1]; kk ++) {
                        derivatives_of_each_layer[jj][mm] += coeff[jj][kk][mm] \
                                                            * temp_derivative_of_input_for_this_layer[kk];
#ifdef DEBUG
                        // printf("derivatives_of_each_layer[%d][%d] = %lf\n", jj, mm, derivatives_of_each_layer[jj][mm]);
                        // printf("coeff[%d][%d][%d] = %lf\n", jj, kk, mm, coeff[jj][kk][mm]);
#endif
                }
            } 
            // FIXME: some problem here      
        }
        else {
            for (int mm = 0; mm < num_of_nodes[jj]; mm ++) {
                derivatives_of_each_layer[jj][mm] = 0;
                for (int kk = 0; kk < num_of_nodes[jj + 1]; kk ++) {
                    if (layer_types[jj] == string("Tanh")) {
                        // printf("tanh\n");
                        derivatives_of_each_layer[jj][mm] += derivatives_of_each_layer[jj + 1][kk] \
                                    * coeff[jj][kk][mm] \
                                    * (1 - output_of_each_layer[jj + 1][kk] * output_of_each_layer[jj + 1][kk]);
                    }
                    else if (layer_types[jj] == string("Linear")) {
                        // printf("linear\n");
                        derivatives_of_each_layer[jj][mm] += derivatives_of_each_layer[jj + 1][kk] \
                                    * coeff[jj][kk][mm] \
                                    * 1;
                    }
                    else {
                        printf("layer type not found!\n\n");
                        return;
                    }
                }
            }
        }
    }
#ifdef DEBUG
    // print out the result for debugging
    // printf("derivatives_of_each_layer = \n");
    // for (int ii = 0; ii < NUM_OF_LAYERS; ii ++) {
    //     printf("layer[%d]: ", ii);
    //     for (int jj = 0; jj < num_of_nodes[ii]; jj ++) {
    //         printf("%lf\t", derivatives_of_each_layer[ii][jj]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
#endif
    return;
}

void ReferenceCalcANN_ForceKernel::get_all_forces_from_derivative_of_first_layer(vector<RealVec>& positionData,
                                                                            vector<RealVec>& forceData,
                                                                            vector<double>& derivatives_of_first_layer) {
    for (int ii = 0; ii < num_of_nodes[0] / 2; ii ++ ) {
        get_force_from_derivative_of_first_layer(2 * ii, 2 * ii + 1, positionData, forceData, derivatives_of_first_layer);
    }
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
    for (int ii = 0; ii < 4; ii ++) {
        idx[ii] = list_of_index_of_atoms_forming_dihedrals[index_of_dihedral][ii];
    }
    

    RealVec diff_1 = positionData[idx[1]] - positionData[idx[0]];
    RealVec diff_2 = positionData[idx[2]] - positionData[idx[1]];
    RealVec diff_3 = positionData[idx[3]] - positionData[idx[2]];

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
            forceData[idx[ii + 1]][jj] += - temp;
#ifdef DEBUG
            // printf("diff_1 = \n");
            // printf("%f\t%f\t%f\n", diff_1[0], diff_1[1], diff_1[2]);
            // printf("temp = %f\n", temp);
            // printf("forceData[%d][%d] = %f\n", idx[ii], jj, forceData[idx[ii]][jj]);
            // printf("forceData[%d][%d] = %f\n", idx[ii + 1], jj, forceData[idx[ii + 1]][jj]);
#endif
        }
    }

#ifdef DEBUG
    // printf("der_of_cos_sin_to_diff = \n");
    // int num_of_rows = 2;
    // int num_of_cols = 9;
    // for (int ii = 0; ii < num_of_rows; ii ++) {
    //     for (int jj = 0; jj < num_of_cols; jj ++) {
    //         printf("%lf\t", der_of_cos_sin_to_diff[ii][jj]);
    //     }
    //     printf("\n");
    // }
#endif

    return;
}

void ReferenceCalcANN_ForceKernel::get_cos_and_sin_of_dihedral_angles(const vector<RealVec>& positionData,
                                                                            vector<RealOpenMM>& cos_sin_value) {
    RealOpenMM temp_cos, temp_sin;
    for (int ii = 0; ii < list_of_index_of_atoms_forming_dihedrals.size(); ii ++) {
        get_cos_and_sin_for_four_atoms(list_of_index_of_atoms_forming_dihedrals[ii][0],
                                       list_of_index_of_atoms_forming_dihedrals[ii][1],
                                       list_of_index_of_atoms_forming_dihedrals[ii][2],
                                       list_of_index_of_atoms_forming_dihedrals[ii][3], 
                                       positionData, temp_cos, temp_sin);
        cos_sin_value.push_back(temp_cos);
        cos_sin_value.push_back(temp_sin);
    }
#ifdef DEBUG
    // printf("cos_sin_value.size() = %d, num_of_nodes[0] = %d\n", cos_sin_value.size(), num_of_nodes[0]);
    assert (cos_sin_value.size() == num_of_nodes[0]);
    assert (cos_sin_value.size() == list_of_index_of_atoms_forming_dihedrals.size() * 2);
#endif
    return;
}

void ReferenceCalcANN_ForceKernel::get_cos_and_sin_for_four_atoms(int idx_1, int idx_2, int idx_3, int idx_4, 
                                const vector<RealVec>& positionData, RealOpenMM& cos_value, RealOpenMM& sin_value) {
    RealVec diff_1 = positionData[idx_2] - positionData[idx_1];
    RealVec diff_2 = positionData[idx_3] - positionData[idx_2];
    RealVec diff_3 = positionData[idx_4] - positionData[idx_3];


    RealVec normal_1 = diff_1.cross(diff_2);
    RealVec normal_2 = diff_2.cross(diff_3);
    normal_1 /= sqrt(normal_1.dot(normal_1));  // normalization
    normal_2 /= sqrt(normal_2.dot(normal_2));
    cos_value = normal_1.dot(normal_2);
    RealVec sin_vec = normal_1.cross(normal_2);
    int sign = (sin_vec[0] + sin_vec[1] + sin_vec[2]) * (diff_2[0] + diff_2[1] + diff_2[2]) > 0 ? 1 : -1;
    sin_value = sqrt(sin_vec.dot(sin_vec)) * sign;
#ifdef DEBUG    
    // printf("positionData[%d] = %f,%f,%f\n", idx_1, positionData[idx_1][0], positionData[idx_1][1],positionData[idx_1][2]);
    // printf("positionData[%d] = %f,%f,%f\n", idx_2, positionData[idx_2][0], positionData[idx_2][1],positionData[idx_2][2]);
    // printf("positionData[%d] = %f,%f,%f\n", idx_3, positionData[idx_3][0], positionData[idx_3][1],positionData[idx_3][2]);
    // printf("positionData[%d] = %f,%f,%f\n", idx_4, positionData[idx_4][0], positionData[idx_4][1],positionData[idx_4][2]);
    // printf("diff_1 = %f,%f,%f\n", diff_1[0], diff_1[1], diff_1[2]);
    // printf("diff_2 = %f,%f,%f\n", diff_2[0], diff_2[1], diff_2[2]);
    // printf("diff_3 = %f,%f,%f\n", diff_3[0], diff_3[1], diff_3[2]);
    // printf("%f,%f,%f\n", normal_1[0], normal_1[1], normal_1[2]);
    // printf("%f,%f,%f\n", normal_2[0], normal_2[1], normal_2[2]);
    // printf("%f,%f,%f\n", sin_vec[0], sin_vec[1], sin_vec[2]);
    // printf("cos_value = %f, sin_value = %f\n", cos_value, sin_value);
#endif
#ifdef DEBUG
    // printf("idx_1 = %d, idx_2 = %d, idx_3 = %d, idx_4 = %d\n", idx_1, idx_2, idx_3, idx_4);
    if (abs(cos_value * cos_value + sin_value * sin_value - 1 ) > 1e-5) {
        printf("cos_value = %lf, sin_value = %lf\n", cos_value, sin_value);
        return;
    }

#endif
    return;
}