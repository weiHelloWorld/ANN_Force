# OpenMM_INSTALL_DIR=/home/fisiksnju/.anaconda2
OpenMM_INSTALL_DIR=/usr/local/openmm
CFLAGS = -g -std=c++11
PLUGIN_DIR=/home/fisiksnju/Dropbox/temp_Linux/ANN_Force
PLUGIN_DIR_INCLUDE=/home/fisiksnju/Dropbox/temp_Linux/ANN_Force/openmmapi/include

FCPPLIBS = -lstdc++

LIB_DIR=$(OpenMM_INSTALL_DIR)/lib
INCLUDE_OPTION=-I$(OpenMM_INSTALL_DIR)/include -I$(PLUGIN_DIR_INCLUDE) -I$(PLUGIN_DIR_INCLUDE)/internal \
-I$(OpenMM_INSTALL_DIR)/include/openmm/reference/ -I/home/fisiksnju/.anaconda2/include/python2.7/\
-I$(PLUGIN_DIR)/platforms/reference -I$(PLUGIN_DIR)/platforms/cuda/include
LIBS= -lOpenMM
LIB_OPTION=-L$(LIB_DIR) $(LIBS)

example: temp_example.cpp
	g++ $(CFLAGS) $(INCLUDE_OPTION) -o example temp_example.cpp $(LIB_OPTION) -L$(LIB_DIR) -lANN

clean:
	rm example

