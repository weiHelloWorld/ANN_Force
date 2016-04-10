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
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
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
    ANN_Force() {};
    
    const std::vector<int>& get_num_of_nodes() const;

    void set_num_of_nodes(std::vector<int> num);

    const std::vector<int>& get_index_of_backbone_atoms() const;

    void set_index_of_backbone_atoms(std::vector<int> indices);

    const std::vector<std::vector<double> >& get_coeffients_of_connections() const;
    
    void set_coeffients_of_connections(std::vector<std::vector<double> > coefficients);

    const std::vector<std::string>& get_layer_types() const;

    void set_layer_types(std::vector<std::string>  temp_layer_types);


    /**
     * Update the per-bond parameters in a Context to match those stored in this Force object.  This method provides
     * an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setBondParameters() to modify this object's parameters, then call updateParametersInContext()
     * to copy them over to the Context.
     * 
     * The only information this method updates is the values of per-bond parameters.  The set of particles involved
     * in a bond cannot be changed, nor can new bonds be added.
     */
    void updateParametersInContext(Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if nonbondedMethod uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const;
};

