#ifndef OPENMM_ANN_FORCE_IMPL_H_
#define OPENMM_ANN_FORCE_IMPL_H_

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

#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include "OpenMM_ANN.h"
#include <utility>
#include <set>
#include <vector>
#include <string>

using std::vector;
using std::map;
using std::string;

namespace OpenMM {


class ANN_ForceImpl : public ForceImpl {
public:
    ANN_ForceImpl(const ANN_Force& owner);
    ~ANN_ForceImpl();
    void initialize(ContextImpl& context);
    const ANN_Force& getOwner() const {
        return owner;
    }
    void updateContextState(ContextImpl& context) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    map<string, double> getDefaultParameters() {
        return map<string, double>(); // This force field doesn't define any parameters.
    }
    vector<string> getKernelNames();

    // vector< pair<int, int> > getBondedParticles() const;
    void updateParametersInContext(ContextImpl& context);
private:
    const ANN_Force& owner;
    Kernel kernel;
};

} // namespace OpenMM

#endif 
