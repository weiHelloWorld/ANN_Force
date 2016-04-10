#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "../include/OpenMM_ANN.h"
#include "ANN_ForceImpl.cpp"

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
	for (int i = 0; i < NUM_OF_BACKBONE_ATOMS; i ++) {
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


ForceImpl* ANN_Force::createImpl() const {
    return new ANN_ForceImpl(*this);
}

void ANN_Force::updateParametersInContext(Context& context) {
    dynamic_cast<ANN_ForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

