#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "../include/openmm/ANN_Force.h"


using namespace OpenMM;
using std::string;
using std::vector;

// TODO

ANN_Force::ANN_Force() {
}

const vector<int>& ANN_Force::get_num_of_nodes() {
    return num_of_nodes;
}

void ANN_Force::set_num_of_nodes(int num[NUM_OF_LAYERS]) {
    for(int i = 0; i < NUM_OF_LAYERS; i ++) {
        num_of_nodes[i] = num[i];
    }
    return;
}

const vector<vector<double> >& ANN_Force::get_coeffients_of_connections() {
    return coeff;
}

void ANN_Force::set_coeffients_of_connections(vector<vector<double> > coefficients) {
    coeff = coefficients;
    return;
}


const vector<string>& ANN_Force::get_layer_types() {
    return layer_types;
}

void ANN_Force::set_layer_types(vector<string>  temp_layer_types) {
    layer_types = temp_layer_types;
    return;
}


ForceImpl* ANN_Force::createImpl() const {
    return new ANN_ForceImpl(*this);
}

void ANN_Force::updateParametersInContext(Context& context) {
    dynamic_cast<ANN_ForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
