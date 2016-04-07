%module ANN

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
%include "std_string.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "OpenMM_ANN.h"

using namespace OpenMM;
%}

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}

#define NUM_OF_LAYERS 3
#define NUM_OF_BACKBONE_ATOMS 60

class ANN_Force : public Force {

public:
    ANN_Force() {};
    
    const vector<int>& get_num_of_nodes() const;

    void set_num_of_nodes(int num[NUM_OF_LAYERS]);

    const vector<int>& get_index_of_backbone_atoms() const;

    void set_index_of_backbone_atoms(int indices[NUM_OF_BACKBONE_ATOMS]);

    const vector<vector<double> >& get_coeffients_of_connections() const;
    
    void set_coeffients_of_connections(vector<vector<double> > coefficients);

    const vector<string>& get_layer_types() const;

    void set_layer_types(vector<string>  temp_layer_types);

    void updateParametersInContext(Context& context);

    bool usesPeriodicBoundaryConditions() const;
};

