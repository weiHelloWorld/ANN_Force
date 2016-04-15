OpenMM_INSTALL_DIR=/usr/local/openmm
PLUGIN_DIR=/home/fisiksnju/Dropbox/temp_Linux/ANN_Force
CFLAGS = -g -std=c++11


FCPPLIBS = -lstdc++

LIB_DIR=$(OpenMM_INSTALL_DIR)/lib
INCLUDE_OPTION=-I$(OpenMM_INSTALL_DIR)/include -I$(OpenMM_INSTALL_DIR)/include/openmm/reference/ -I$(PLUGIN_DIR)/openmmapi/include -I/home/fisiksnju/.anaconda2/include/python2.7/
LIBS= -lOpenMM
LIB_OPTION=-L$(LIB_DIR) $(LIBS)

test_ANN_Force:
	g++ $(CFLAGS) $(INCLUDE_OPTION) -o test_ANN_Force test_ANN_Force.cpp $(LIB_OPTION)

test_ANN_package:
	g++ $(CFLAGS) $(INCLUDE_OPTION) -o test_ANN_package test_ANN_package.cpp $(LIB_OPTION)

package:
	g++ $(CFLAGS) $(INCLUDE_OPTION) -shared -o libANN.so -fPIC test_ANN_package.cpp

wrapper:
	swig -c++ -python ANN.i && \
	g++ $(CFLAGS) $(INCLUDE_OPTION) \
	-shared -fPIC -c ANN_wrap.cxx test_ANN_package.cpp && \
	g++ -shared $(INCLUDE_OPTION) *.o -o _ANN.so \
	$(LIB_OPTION) -L$(PLUGIN_DIR)/openmmapi/tests/ -lANN && \
	rm *.o

clean:
	rm test_ANN_Force test_ANN_package _ANN.so libANN.so *.o ANN_wrap.cxx ANN.py* 
