%module ANN

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

/*
 * The following lines are needed to handle vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
%include "std_string.i"
namespace std {
  %template(double_vector) vector<double>;
  %template(int_vector) vector<int>;
  %template(d_double_vector) vector<vector<double> >;
  %template(d_int_vector) vector<vector<int> >;
  %template(string_vector) vector<string>;
};

%{
#include "OpenMM_ANN.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"

using namespace OpenMM;
%}

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}

#define NUM_OF_LAYERS 3
#define NUM_OF_BACKBONE_ATOMS 60

class ANN_Force : public OpenMM::Force {

public:
    const std::vector<int>& get_num_of_nodes() const;

    void set_num_of_nodes(std::vector<int> num);

    const std::vector<int>& get_index_of_backbone_atoms() const;

    void set_index_of_backbone_atoms(std::vector<int> indices);

    const std::vector<std::vector<double> >& get_coeffients_of_connections() const;
    
    void set_coeffients_of_connections(std::vector<std::vector<double> > coefficients);

    const std::vector<std::string>& get_layer_types() const;

    void set_layer_types(std::vector<std::string>  temp_layer_types);

    const std::vector<std::vector<double> >& get_values_of_biased_nodes() const;

    void set_values_of_biased_nodes(std::vector<std::vector<double> > bias);

    const std::vector<double>& get_potential_center() const;

    void set_potential_center(std::vector<double> temp_potential_center);

    const double get_force_constant() const;

    void set_force_constant(double temp_force_constant);

    const int get_data_type_in_input_layer() const;

    void set_data_type_in_input_layer(int temp_data_type_in_input_layer);

    const std::vector<std::vector<int> >& get_list_of_index_of_atoms_forming_dihedrals() const;

    void set_list_of_index_of_atoms_forming_dihedrals(std::vector<std::vector<int> > temp_list_of_index);    
    
    void set_list_of_index_of_atoms_forming_dihedrals_from_index_of_backbone_atoms(std::vector<int> index_of_backbone_atoms);

};

