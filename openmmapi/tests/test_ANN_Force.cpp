#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/OpenMMException.h"

#include "../include/OpenMM_ANN.h"
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
    cout << "running test_setters_getters_1\n";
    ANN_Force my_force;
    int num_of_nodes[NUM_OF_LAYERS] = {76, 10, 2};
    my_force.set_num_of_nodes(num_of_nodes);
    for(int i = 0; i < NUM_OF_LAYERS; i ++) {
        ASSERT_EQUAL_TOL(num_of_nodes[i], my_force.get_num_of_nodes()[i], TOL); 
    }
    return;
}

static void test_setters_getters_2() {
    cout << "running test_setters_getters_2\n";
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

static void test_setters_getters_3() {
    cout << "running test_setters_getters_3\n";
    ANN_Force my_force;
    int idx[60] = {1, 2, 3, 17, 18, 19, 36, 37, 38, 57, 58, 59, 76, 77, 78, 93, 94, 95, \
        117, 118, 119, 136, 137, 138, 158, 159, 160, 170, 171, 172, 177, 178, 179, 184, \
        185, 186, 198, 199, 200, 209, 210, 211, 220, 221, 222, 227, 228, 229, 251, 252, \
        253, 265, 266, 267, 279, 280, 281, 293, 294, 295};
    my_force.set_index_of_backbone_atoms(idx);
    auto output = my_force.get_index_of_backbone_atoms();
    for (int i = 0; i < 60; i ++) {
        assert(output[i] == idx[i]);
    }
    return;
}




int main(int numberOfArguments, char* argv[]) {

    try {
        test_setters_getters_1();
        test_setters_getters_2();
        test_setters_getters_3();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
