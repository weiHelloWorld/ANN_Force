#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/System.h"

#include "../include/openmm/ANN_Force.h"
#include "../src/ANN_Force.cpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace OpenMM;
using std::string;
using std::vector;

const double TOL = 1e-5;

static void test_setters_getters() {
    ANN_Force my_force;
    int num_of_nodes[NUM_OF_LAYERS] = {76, 10, 2};
    my_force.set_num_of_nodes(num_of_nodes);
    for(int i = 0; i < NUM_OF_LAYERS; i ++) {
        ASSERT_EQUAL_TOL(num_of_nodes[i], my_force.get_num_of_nodes()[i], TOL);    
    }
    return;
}


int main(int numberOfArguments, char* argv[]) {

    try {
        std::cout << "running test for ANN_Force..." << std::endl;
        // test_setters_getters();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
