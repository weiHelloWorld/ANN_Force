/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaANN_Kernels.h"
#include "CudaANN_KernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"

using namespace OpenMM;
using namespace std;

class CudaANN_ForceInfo : public CudaForceInfo {   // TODO: what to do with this part?
public:
    CudaANN_ForceInfo(const ANN_Force& force) : force(force) {
    }
    int getNumParticleGroups() {
        // return force.getNumBonds();
        return 1;
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        // int particle1, particle2;
        // double length, k;
        // force.getBondParameters(index, particle1, particle2, length, k);
        // particles.resize(2);
        // particles[0] = particle1;
        // particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        // int particle1, particle2;
        // double length1, length2, k1, k2;
        // force.getBondParameters(group1, particle1, particle2, length1, k1);
        // force.getBondParameters(group2, particle1, particle2, length2, k2);
        // return (length1 == length2 && k1 == k2);
        return true;
    }
private:
    const ANN_Force& force;
};

CudaCalcANN_ForceKernel::~CudaCalcANN_ForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
    // TODO: delete other parameters
}

void CudaCalcANN_ForceKernel::initialize(const System& system, const ANN_Force& force) {
    cu.setAsCurrent();
    
    data_type_in_input_layer = force.get_data_type_in_input_layer();
    if (data_type_in_input_layer != 1) {
        throw OpenMMException("not yet implemented for data_type_in_input_layer = " 
            + std::to_string(data_type_in_input_layer) + "\n");
    }
    if (NUM_OF_LAYERS != 3) {
        throw OpenMMException("not yet implemented for NUM_OF_LAYERS != " + std::to_string(NUM_OF_LAYERS) + "\n");
    }
    
    // for writing cuda code: 
    // 1. store these parameters in device memory
    // 2. set up replacement text
    // 3. write .cu file

    // num_of_nodes = CudaArray::create<int>(cu, NUM_OF_LAYERS, "num_of_nodes_cuda");
    // num_of_nodes -> upload(force.get_num_of_nodes());

    int num_of_backbone_atoms = force.get_index_of_backbone_atoms().size();
    int num_of_parallel_threads = 60;    // FIXME: there should be better way to determine this value
    vector<vector<int> > index_of_atoms_in_the_force(num_of_parallel_threads);
    for (int jj = 0; jj < num_of_parallel_threads; jj ++) {
        index_of_atoms_in_the_force[jj] = force.get_index_of_backbone_atoms();
        for (int ii = 0; ii < num_of_backbone_atoms; ii ++) {
            index_of_atoms_in_the_force[jj][ii] -= 1;  // because in pdb file, index start with 1
        }
    }

    // convert to CUDA array
    auto temp_num_of_nodes = force.get_num_of_nodes();
    num_of_nodes = convert_vector_to_CudaArray(force.get_num_of_nodes(), "1");
    index_of_backbone_atoms = convert_vector_to_CudaArray(force.get_index_of_backbone_atoms(), "3");
    {
        map<string, int> temp_replacement_layer_types;  // since string is not supported in CUDA kernel
        temp_replacement_layer_types["Linear"] = 0; temp_replacement_layer_types["Tanh"] = 1; temp_replacement_layer_types["Circular"] = 2; 
        vector<int> temp_layer_types(NUM_OF_LAYERS - 1);
        for (int ii = 0; ii < NUM_OF_LAYERS - 1; ii ++) {
            temp_layer_types[ii] = temp_replacement_layer_types[force.get_layer_types()[ii]];
        }
        layer_types = convert_vector_to_CudaArray(temp_layer_types, "4");
    }
    potential_center = convert_vector_to_CudaArray(convert_vector_of_type_to_float(force.get_potential_center()), "5");
    {
        vector<float> temp_force_constant({(float) force.get_force_constant()});
        force_constant = convert_vector_to_CudaArray(temp_force_constant, "6");
    }
    {
        vector<float> temp_scaling_factor({(float) force.get_scaling_factor()});
        scaling_factor = convert_vector_to_CudaArray(temp_scaling_factor, "7");   
    }
    {
        vector<float> temp_input_0(temp_num_of_nodes[0]), temp_input_1(temp_num_of_nodes[1]), temp_input_2(temp_num_of_nodes[2]);
        vector<float> temp_output_0(temp_num_of_nodes[0]), temp_output_1(temp_num_of_nodes[1]), temp_output_2(temp_num_of_nodes[2]);
        input_0 = convert_vector_to_CudaArray(temp_input_0, "input_0");
        input_1 = convert_vector_to_CudaArray(temp_input_1, "input_1");
        input_2 = convert_vector_to_CudaArray(temp_input_2, "input_2");
        // output_0 = convert_vector_to_CudaArray(temp_output_0, "output_0");
        // output_1 = convert_vector_to_CudaArray(temp_output_1, "output_1");
        // output_2 = convert_vector_to_CudaArray(temp_output_2, "output_2");
    }
    {
        // cout << "temp" << endl;
        // auto temp = convert_vector_of_type_to_float(force.get_coeffients_of_connections()[0]);
        // for (auto item: temp) {
        //     cout << item << "\t";
        // }
        coeff_0 = convert_vector_to_CudaArray(convert_vector_of_type_to_float(force.get_coeffients_of_connections()[0]), "coeff_0");
        coeff_1 = convert_vector_to_CudaArray(convert_vector_of_type_to_float(force.get_coeffients_of_connections()[1]), "coeff_1");
        bias_0 = convert_vector_to_CudaArray(convert_vector_of_type_to_float(force.get_values_of_biased_nodes()[0]), "bias_0");
        bias_1 = convert_vector_to_CudaArray(convert_vector_of_type_to_float(force.get_values_of_biased_nodes()[1]), "bias_1");
        // float temp[50];
        // get_data_from_CudaArray(coeff_0, temp);
        // for (int ii = 0; ii < 50; ii ++) {
        //     cout << temp[ii] << "\t";
        // }
    }

    
    // replace text in .cu file
    map<string, string> replacements;  
    replacements["POTENTIAL_CENTER"] = cu.getBondedUtilities().addArgument(potential_center->getDevicePointer(), "float");
    replacements["FORCE_CONSTANT"] = cu.getBondedUtilities().addArgument(force_constant->getDevicePointer(), "float");
    replacements["SCALING_FACTOR"] = cu.getBondedUtilities().addArgument(scaling_factor->getDevicePointer(), "float"); 
    replacements["NUM_OF_NODES"] = cu.getBondedUtilities().addArgument(num_of_nodes->getDevicePointer(), "int"); 
    replacements["LAYER_TYPES"] = cu.getBondedUtilities().addArgument(layer_types->getDevicePointer(), "int"); 
    replacements["INPUT_0"] = cu.getBondedUtilities().addArgument(input_0->getDevicePointer(), "float"); 
    replacements["INPUT_1"] = cu.getBondedUtilities().addArgument(input_1->getDevicePointer(), "float"); 
    replacements["INPUT_2"] = cu.getBondedUtilities().addArgument(input_2->getDevicePointer(), "float"); 
    // replacements["OUTPUT_0"] = cu.getBondedUtilities().addArgument(output_0->getDevicePointer(), "float"); 
    // replacements["OUTPUT_1"] = cu.getBondedUtilities().addArgument(output_1->getDevicePointer(), "float"); 
    // replacements["OUTPUT_2"] = cu.getBondedUtilities().addArgument(output_2->getDevicePointer(), "float"); 
    replacements["COEFF_0"] = cu.getBondedUtilities().addArgument(coeff_0->getDevicePointer(), "float"); 
    replacements["COEFF_1"] = cu.getBondedUtilities().addArgument(coeff_1->getDevicePointer(), "float"); 
    replacements["BIAS_0"] = cu.getBondedUtilities().addArgument(bias_0->getDevicePointer(), "float"); 
    replacements["BIAS_1"] = cu.getBondedUtilities().addArgument(bias_1->getDevicePointer(), "float"); 

    // preprocessing for source code
    auto source_code_for_force_before_replacement = CudaANN_KernelSources::ANN_Force;
    assert (force.get_index_of_backbone_atoms().size() * 3 == temp_num_of_nodes[0]);
    stringstream temp_string;
    // temp_string << "int num_of_parallel_threads = " << num_of_parallel_threads << ";\n";
    // temp_string << "int num_of_rows, num_of_cols;\n";
    temp_string << "float force_constant = " << force.get_force_constant() << ";\n";
    
    for (int ii = 0; ii < num_of_backbone_atoms; ii ++) {
        temp_string << "INPUT_0[" << (3 * ii + 0) << "] = pos" << (ii + 1) << ".x / " << force.get_scaling_factor() << ";\n";
        temp_string << "INPUT_0[" << (3 * ii + 1) << "] = pos" << (ii + 1) << ".y / " << force.get_scaling_factor() << ";\n";
        temp_string << "INPUT_0[" << (3 * ii + 2) << "] = pos" << (ii + 1) << ".z / " << force.get_scaling_factor() << ";\n";
    }
    if (remove_translation_degrees_of_freedom) {
        assert (num_of_parallel_threads >= 3);
        temp_string << "float coor_center_of_mass = 0;\n";  // actually coor_center_of_mass is one component, each thread only needs to handle one component
        temp_string << "if (index < 3) {\n";
        temp_string << "    for (int ii = index; ii < " << 3 * num_of_backbone_atoms << "; ii += 3) {\n";
        temp_string << "        coor_center_of_mass += INPUT_0[ii];\n";
        temp_string << "    }\n";
        temp_string << "    coor_center_of_mass /= " << num_of_backbone_atoms << ";\n";
        temp_string << "    for (int ii = index; ii < " << 3 * num_of_backbone_atoms << "; ii += 3) {\n";
        temp_string << "        INPUT_0[ii] -= coor_center_of_mass;\n";
        temp_string << "    }\n";
        temp_string << "}\n";
        // temp_string << "__syncthreads();\n";
    }
    temp_string << "\n"; 
    temp_string << "__syncthreads();\n";
    temp_string << "// forward propagation\n";
    for (int ii = 0; ii < NUM_OF_LAYERS - 1; ii ++) {

        temp_string << "for (int ii = index; ii < " << (temp_num_of_nodes[ii + 1]) << "; ii += " << num_of_parallel_threads << ") {\n";
        temp_string << "    float temp = BIAS_" << (ii) << "[ii];\n";
        temp_string << "    for (int jj = 0; jj < " << (temp_num_of_nodes[ii]) << "; jj ++) {\n";
        temp_string << "        temp += COEFF_" << (ii) << "[ii * " << (temp_num_of_nodes[ii]) << " + jj] * INPUT_" << (ii) << "[jj];\n";
        temp_string << "    }\n";
        if (force.get_layer_types()[ii] == "Tanh") {
            temp_string << "    INPUT_" << (ii + 1) << "[ii] = tanh(temp);\n";
        }
        else {
            temp_string << "    INPUT_" << (ii + 1) << "[ii] = temp;\n";
        }
        temp_string << "}\n";
        temp_string << "__syncthreads();\n";
    }
    temp_string << "// backward propagation, INPUT_{0,1,2} are reused to store derivatives in each layer\n";
    int dim_of_PC_space = temp_num_of_nodes[NUM_OF_LAYERS - 1];
    temp_string << "for (int ii = index; ii < " << dim_of_PC_space << "; ii += " << num_of_parallel_threads << ") {\n";
    temp_string << "    float temp = INPUT_" << (NUM_OF_LAYERS - 1) << "[ii];\n";
    temp_string << "    energy += 0.5 * (temp - POTENTIAL_CENTER[ii]) * (temp - POTENTIAL_CENTER[ii]) * force_constant;\n";
    if (force.get_layer_types()[NUM_OF_LAYERS - 2] == "Tanh") {
        temp_string << "    INPUT_" << (NUM_OF_LAYERS - 1) << "[ii] = (temp - POTENTIAL_CENTER[ii]) * force_constant * (1 - temp * temp);\n";    
    }
    else if (force.get_layer_types()[NUM_OF_LAYERS - 2] == "Linear") {
        temp_string << "    INPUT_" << (NUM_OF_LAYERS - 1) << "[ii] = (temp - POTENTIAL_CENTER[ii]) * force_constant;\n";    
    }
    temp_string << "}\n";
    temp_string << "__syncthreads();\n";
    for (int ii = NUM_OF_LAYERS - 1; ii > 0; ii --) {
        temp_string << "for (int ii = index; ii < " << temp_num_of_nodes[ii - 1] << "; ii += " << num_of_parallel_threads << ") {\n";
        temp_string << "    float temp = 0;\n";
        temp_string << "    for (int jj = 0; jj < " << temp_num_of_nodes[ii] << "; jj ++) {\n";
        temp_string << "        temp += COEFF_" << (ii - 1) << "[ii + jj * " << temp_num_of_nodes[ii - 1] << "] * INPUT_" << ii << "[jj];\n";
        temp_string << "    }\n";
        if ((ii - 2 >= 0) && (force.get_layer_types()[ii - 2] == "Tanh")) {
            temp_string << "    float temp_" << (ii - 1) << " = INPUT_" << (ii - 1) << "[ii];\n";
            temp_string << "    INPUT_" << (ii - 1) << "[ii] = temp * (1 - temp_" << (ii - 1) << " * temp_" << (ii - 1) << ");\n";
        }
        else {
            temp_string << "    INPUT_" << (ii - 1) << "[ii] = temp;\n";
        }
        temp_string << "}\n";
        temp_string << "__syncthreads();\n";
    }
    
    // source_code_for_force_before_replacement = temp_string + source_code_for_force_before_replacement;

    temp_string << "\n";
    temp_string << "real3 average_deriv_on_input_layer = make_real3(0.0);\n";
    if (remove_translation_degrees_of_freedom) {
        temp_string << "for (int ii = 0; ii < " << num_of_backbone_atoms << "; ii ++ ) {\n";
        temp_string << "    average_deriv_on_input_layer.x += INPUT_0[3 * ii + 0];\n";
        temp_string << "    average_deriv_on_input_layer.y += INPUT_0[3 * ii + 1];\n";
        temp_string << "    average_deriv_on_input_layer.z += INPUT_0[3 * ii + 2];\n";
        temp_string << "}\n";
        temp_string << "average_deriv_on_input_layer = average_deriv_on_input_layer / " << num_of_backbone_atoms << ";\n";
    }
    for (int ii = 0; ii < num_of_backbone_atoms; ii ++) {
        temp_string << "real3 force" << (ii + 1) << ";\n";
    } 
    temp_string << "\n";
    temp_string << "if (index == 0) { \n";
    for (int ii = 0; ii < num_of_backbone_atoms; ii ++) {  // only thread 0 calculate force, avoid repeated computation
        temp_string << "    force" << (ii + 1) << " = make_real3( ";
        temp_string << "- (INPUT_0[" << (3 * ii + 0) << "] - average_deriv_on_input_layer.x) / " << force.get_scaling_factor() << ", ";
        temp_string << "- (INPUT_0[" << (3 * ii + 1) << "] - average_deriv_on_input_layer.y) / " << force.get_scaling_factor() << ", ";
        temp_string << "- (INPUT_0[" << (3 * ii + 2) << "] - average_deriv_on_input_layer.z) / " << force.get_scaling_factor() << ") ;\n";
    }
    temp_string << "}\nelse { \n";
    for (int ii = 0; ii < num_of_backbone_atoms; ii ++) {
        temp_string << "    force" << (ii + 1) << " = make_real3(0.0);\n";
    } 
    temp_string << "}\n";
    source_code_for_force_before_replacement = temp_string.str();
    // source_code_for_force_before_replacement += temp_string;
    auto source_code_for_force_after_replacement = cu.replaceStrings(source_code_for_force_before_replacement, replacements);

    cu.getBondedUtilities().addInteraction(index_of_atoms_in_the_force, source_code_for_force_after_replacement , force.getForceGroup());
#ifdef BUDEG_CUDA
    cout << "before replacement:\n" << source_code_for_force_before_replacement << endl;
    // cout << "after replacement:\n" << source_code_for_force_after_replacement << endl; 
#endif
    cu.addForce(new CudaANN_ForceInfo(force));
}

double CudaCalcANN_ForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CudaCalcANN_ForceKernel::copyParametersToContext(ContextImpl& context, const ANN_Force& force) {
    // cu.setAsCurrent();
    // int numContexts = cu.getPlatformData().contexts.size();
    // int startIndex = cu.getContextIndex()*force.getNumBonds()/numContexts;
    // int endIndex = (cu.getContextIndex()+1)*force.getNumBonds()/numContexts;
    // if (numBonds != endIndex-startIndex)
    //     throw OpenMMException("updateParametersInContext: The number of bonds has changed");
    // if (numBonds == 0)
    //     return;
    
    // // Record the per-bond parameters.
    
    // vector<float2> paramVector(numBonds);
    // for (int i = 0; i < numBonds; i++) {
    //     int atom1, atom2;
    //     double length, k;
    //     force.getBondParameters(startIndex+i, atom1, atom2, length, k);
    //     paramVector[i] = make_float2((float) length, (float) k);
    // }
    // params->upload(paramVector);
    
    // // Mark that the current reordering may be invalid.
    
    // cu.invalidateMolecules();
}
