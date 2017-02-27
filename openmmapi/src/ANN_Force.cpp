#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "../include/OpenMM_ANN.h"
// #include "ANN_ForceImpl.cpp"

using namespace OpenMM;



const std::vector<int>& ANN_Force::get_num_of_nodes() const {
    return num_of_nodes;
}

void ANN_Force::set_num_of_nodes(std::vector<int> num) {
    for(int i = 0; i < NUM_OF_LAYERS; i ++) {
        num_of_nodes[i] = num[i];
    }
    return;
}

const std::vector<int>& ANN_Force::get_index_of_backbone_atoms()const {
	return index_of_backbone_atoms;
}

void ANN_Force::set_index_of_backbone_atoms(std::vector<int> indices){
    index_of_backbone_atoms.resize(indices.size());
	for (int i = 0; i < indices.size(); i ++) {
		index_of_backbone_atoms[i] = indices[i];
	}
	return;
}

const std::vector<std::vector<double> >& ANN_Force::get_coeffients_of_connections() const {
    return coeff;
}

void ANN_Force::set_coeffients_of_connections(std::vector<std::vector<double> > coefficients) {
    coeff = coefficients;
    return;
}


const std::vector<std::string>& ANN_Force::get_layer_types() const {
    return layer_types;
}

void ANN_Force::set_layer_types(std::vector<std::string>  temp_layer_types) {
    layer_types = temp_layer_types;
    return;
}

const std::vector<std::vector<double> >& ANN_Force::get_values_of_biased_nodes() const {
    return values_of_biased_nodes;
}

void ANN_Force::set_values_of_biased_nodes(std::vector<std::vector<double> > bias) {
    values_of_biased_nodes = bias;
    return;
}

const std::vector<double>& ANN_Force::get_potential_center() const {
    return potential_center;
}

void ANN_Force::set_potential_center(std::vector<double> temp_potential_center) {
    potential_center = temp_potential_center;
    return;
}

const double ANN_Force::get_force_constant() const {
    return force_constant;
}

void ANN_Force::set_force_constant(double temp_force_constant) {
    force_constant = temp_force_constant;
    return;
}

const double ANN_Force::get_scaling_factor() const {
    return scaling_factor;
}

void ANN_Force::set_scaling_factor(double temp_scaling_factor) {
    scaling_factor = temp_scaling_factor;
    return;
}

const int ANN_Force::get_data_type_in_input_layer() const {
    return data_type_in_input_layer;
}

void ANN_Force::set_data_type_in_input_layer(int temp_data_type_in_input_layer) {
    data_type_in_input_layer = temp_data_type_in_input_layer;
    return;
}

const std::vector<std::vector<int> >& ANN_Force::get_list_of_index_of_atoms_forming_dihedrals() const {
    return list_of_index_of_atoms_forming_dihedrals;
}

void ANN_Force::set_list_of_index_of_atoms_forming_dihedrals(std::vector<std::vector<int> > temp_list_of_index) {
    num_of_dihedrals = temp_list_of_index.size();
    for (int ii = 0; ii < num_of_dihedrals; ii ++) {
        list_of_index_of_atoms_forming_dihedrals.push_back(std::vector<int>());
        for (int jj = 0; jj < 4; jj ++) {
            list_of_index_of_atoms_forming_dihedrals[ii].push_back(temp_list_of_index[ii][jj] - 1);
            // should "-1", because in PDB file, the index starts from 1
        }
    }
    return;
}

void ANN_Force::set_list_of_index_of_atoms_forming_dihedrals_from_index_of_backbone_atoms(std::vector<int>\
                                                                                 index_of_backbone_atoms) {
    int num_of_residues = index_of_backbone_atoms.size() / 3;
    num_of_dihedrals = num_of_residues * 2 - 2;
    
    int count = 0;
    for (int ii = 0; ii < num_of_residues; ii ++) {
        if (ii != 0) {
            list_of_index_of_atoms_forming_dihedrals.push_back(std::vector<int>());
            list_of_index_of_atoms_forming_dihedrals[count].push_back(index_of_backbone_atoms[3 * ii - 1] - 1);
            list_of_index_of_atoms_forming_dihedrals[count].push_back(index_of_backbone_atoms[3 * ii + 0] - 1);
            list_of_index_of_atoms_forming_dihedrals[count].push_back(index_of_backbone_atoms[3 * ii + 1] - 1);
            list_of_index_of_atoms_forming_dihedrals[count].push_back(index_of_backbone_atoms[3 * ii + 2] - 1);
            count ++;
        }
        if (ii != num_of_residues - 1) {
            list_of_index_of_atoms_forming_dihedrals.push_back(std::vector<int>());
            list_of_index_of_atoms_forming_dihedrals[count].push_back(index_of_backbone_atoms[3 * ii + 0] - 1);
            list_of_index_of_atoms_forming_dihedrals[count].push_back(index_of_backbone_atoms[3 * ii + 1] - 1);
            list_of_index_of_atoms_forming_dihedrals[count].push_back(index_of_backbone_atoms[3 * ii + 2] - 1);
            list_of_index_of_atoms_forming_dihedrals[count].push_back(index_of_backbone_atoms[3 * ii + 3] - 1);
            count ++;
        }
    }
#ifdef DEBUG
    assert (count == num_of_dihedrals);
    // printf("list_of_index_of_atoms_forming_dihedrals = \n");
    // for (int ii = 0; ii < num_of_dihedrals; ii ++) {
    //     for (int jj = 0; jj < 4; jj ++) {
    //         printf("%d\t", list_of_index_of_atoms_forming_dihedrals[ii][jj]);
    //     }
    //     printf("\n");
    // }
#endif
    return;
}

ForceImpl* ANN_Force::createImpl() const {
    return new ANN_ForceImpl(*this);
}

void ANN_Force::updateParametersInContext(Context& context) {
    dynamic_cast<ANN_ForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

