OpenMM_INSTALL_DIR=/usr/local/openmm
PLUGIN_DIR=/home/fisiksnju/Dropbox/temp_Linux/ANN_Force
CFLAGS = -g -std=c++11


FCPPLIBS = -lstdc++

LIB_DIR=$(OpenMM_INSTALL_DIR)/lib
INCLUDE_OPTION=-I$(OpenMM_INSTALL_DIR)/include -I$(OpenMM_INSTALL_DIR)/include/openmm/reference/ -I$(PLUGIN_DIR)/openmmapi/include -I/home/fisiksnju/.anaconda2/include/python2.7/
LIBS= -lOpenMM
LIB_OPTION=-L$(LIB_DIR) $(LIBS)

example_1: example_1.cpp
	g++ $(CFLAGS) $(INCLUDE_OPTION) -o example_1 example_1.cpp $(LIB_OPTION)

clean:
	rm example_1