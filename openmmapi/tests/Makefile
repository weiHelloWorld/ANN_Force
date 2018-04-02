HOME_DIR=/home/kengyangyao
# SOURCE_DIR=/home/weichen9/weichen9/to_cluster/ANN_Force
SOURCE_DIR=$(HOME_DIR)/Dropbox/temp_Linux/ANN_Force

OpenMM_INSTALL_DIR=$(HOME_DIR)/.openmm/include
PYTHON_PACKAGE_DIR=$(HOME_DIR)/.anaconda2/lib/python2.7/site-packages

CFLAGS = -g -std=c++11 -Wall -Wextra -fsanitize=address

FCPPLIBS = -lstdc++

LIB_DIR=$(HOME_DIR)/.openmm/lib
PLUGIN_DIR=$(LIB_DIR)/plugins
INCLUDE_OPTION=-I$(OpenMM_INSTALL_DIR) -I$(OpenMM_INSTALL_DIR)/openmm/reference/ -I$(OpenMM_INSTALL_DIR)/openmm/internal -I$(SOURCE_DIR)/openmmapi/include -I$(HOME_DIR)/.anaconda2/include/python2.7/
LIB_OPTION=-L$(LIB_DIR) -lOpenMM -L$(LIB_DIR) -lANN -L$(PLUGIN_DIR) -lANNReference

test_ANN_package: test_ANN_package.cpp
	g++ $(CFLAGS) $(INCLUDE_OPTION) -o test_ANN_package test_ANN_package.cpp $(LIB_OPTION)

	
clean:
	rm test_ANN_package
