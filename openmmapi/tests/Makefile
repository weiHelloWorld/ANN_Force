OpenMM_INSTALL_DIR=/usr/local/openmm
PLUGIN_DIR=/home/fisiksnju/Dropbox/temp_Linux/ANN_Force
CFLAGS = -g -std=c++11


FCPPLIBS = -lstdc++

LIB_DIR=$(OpenMM_INSTALL_DIR)/lib
INCLUDE_OPTION=-I$(OpenMM_INSTALL_DIR)/include -I$(OpenMM_INSTALL_DIR)/include/openmm/reference/ -I$(PLUGIN_DIR)/openmmapi/include
LIBS= -lOpenMM
LIB_OPTION=-L$(LIB_DIR) $(LIBS)

test_ANN_Force:  test_ANN_Force.cpp
	g++ $(CFLAGS) $(INCLUDE_OPTION) -o test_ANN_Force test_ANN_Force.cpp $(LIB_OPTION)

test_ANN_package: test_ANN_package.cpp
	g++ $(CFLAGS) $(INCLUDE_OPTION) -o test_ANN_package test_ANN_package.cpp $(LIB_OPTION)

clean:
	rm test_ANN_Force test_ANN_package
