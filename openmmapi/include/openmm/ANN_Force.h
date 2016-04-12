#ifndef OPENMM_ANN_FORCE_H_
#define OPENMM_ANN_FORCE_H_

/* -------------------------------------------------------------------------- *
 *                                                                            *
 * Authors: Wei Chen  @ UIUC 
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

#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include "openmm/OpenMMException.h"
#include <map>
#include <vector>
#include <string>


#define NUM_OF_LAYERS 3  
// this is the number of layers associated with mapping from data in original space to 
// principal component space.  If the value is 3, the corresponding ANN network is 5-layer.

#define NUM_OF_BACKBONE_ATOMS 60

namespace OpenMM {

/**
 * TODO: description later
 */

class ANN_Force : public OpenMM::Force {

public:
    ANN_Force() {};
    ~ANN_Force() {};
    
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
    bool usesPeriodicBoundaryConditions() const {
        return false;
    }
protected:
    // double _globalQuarticK, _globalCubicK;
    OpenMM::ForceImpl* createImpl() const;
private:
    std::vector<int> num_of_nodes = std::vector<int>(NUM_OF_LAYERS);    // store the number of nodes for first 3 layers
    std::vector<int> index_of_backbone_atoms = std::vector<int>(NUM_OF_BACKBONE_ATOMS); 
    std::vector<std::vector<double> > coeff = std::vector<std::vector<double> >(NUM_OF_LAYERS - 1);  // TODO: use better implementations?
    std::vector<std::string> layer_types = std::vector<std::string>(NUM_OF_LAYERS - 1); // the input layer is not included
    std::vector<std::vector<double> > values_of_biased_nodes = std::vector<std::vector<double> >(NUM_OF_LAYERS - 1);
};


} // namespace OpenMM

#endif /*OPENMM_ANN_FORCE_H_*/
