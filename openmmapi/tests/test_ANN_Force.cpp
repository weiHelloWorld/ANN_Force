#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/OpenMMException.h"

#include "../include/openmm/ANN_Force.h"
#include "../src/ANN_Force.cpp"

#include <iostream>
#include <vector>
#include <math.h>
#include <cassert>

using namespace OpenMM;
using std::string;
using std::vector;
using std::cout;
using std::endl;

const double TOL = 1e-5;

static void test_setters_getters_1() {
    ANN_Force my_force;
    int num_of_nodes[NUM_OF_LAYERS] = {76, 10, 2};
    my_force.set_num_of_nodes(num_of_nodes);
    for(int i = 0; i < NUM_OF_LAYERS; i ++) {
        ASSERT_EQUAL_TOL(num_of_nodes[i], my_force.get_num_of_nodes()[i], TOL); 
    }
    return;
}

static void test_setters_getters_2() {
    ANN_Force my_force;
    vector<string> types;
    types.push_back("Tanh1");
    types.push_back("Tanh2");
    types.push_back("Tanh3");
    my_force.set_layer_types(types);
    auto out_types = my_force.get_layer_types();
    for(int i = 0; i < 3; i ++) {
        // cout << out_types[i] << endl;
        assert(out_types[i].compare(types[i]) == 0);
    }
    return;

}



int main(int numberOfArguments, char* argv[]) {

    try {
        std::cout << "running test for ANN_Force..." << std::endl;
        test_setters_getters_1();
        test_setters_getters_2();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
