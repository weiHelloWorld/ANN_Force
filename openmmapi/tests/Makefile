OpenMM_INSTALL_DIR=/usr/local/openmm
SOURCE_DIR=/home/fisiksnju/Dropbox/temp_Linux/ANN_Force
PYTHON_PACKAGE_DIR=/home/fisiksnju/.anaconda2/lib/python2.7/site-packages
CFLAGS = -g -std=c++11

FCPPLIBS = -lstdc++

LIB_DIR=$(OpenMM_INSTALL_DIR)/lib
PLUGIN_DIR=$(LIB_DIR)/plugins
INCLUDE_OPTION=-I$(OpenMM_INSTALL_DIR)/include -I$(OpenMM_INSTALL_DIR)/include/openmm/reference/ -I$(SOURCE_DIR)/openmmapi/include -I/home/fisiksnju/.anaconda2/include/python2.7/
LIBS= -lOpenMM
LIB_OPTION=-L$(LIB_DIR) $(LIBS)

test_ANN_Force:
	g++ $(CFLAGS) $(INCLUDE_OPTION) -o test_ANN_Force test_ANN_Force.cpp $(LIB_OPTION)

test_ANN_package:
	g++ $(CFLAGS) $(INCLUDE_OPTION) -o test_ANN_package test_ANN_package.cpp $(LIB_OPTION)

package:
	g++ $(CFLAGS) $(INCLUDE_OPTION) -shared -o libANN.so -fPIC test_ANN_package.cpp

wrapper: package
	swig -c++ -python ANN.i && \
	g++ $(CFLAGS) $(INCLUDE_OPTION) \
	-shared -fPIC -c ANN_wrap.cxx test_ANN_package.cpp && \
	g++ -shared $(INCLUDE_OPTION) *.o -o _ANN.so \
	$(LIB_OPTION) -L$(SOURCE_DIR)/openmmapi/tests/ -lANN && \
	rm *.o

install: wrapper
	sudo mv libANN.so $(PLUGIN_DIR) && \
	mv _ANN.so $(PYTHON_PACKAGE_DIR) && \
	echo "========================================" &&\
	echo "please add $(PLUGIN_DIR) and related directories into LD_LIBRARY_PATH, an example is: export LD_LIBRARY_PATH=/usr/local/openmm/lib:/home/fisiksnju/Dropbox/temp_Linux/ANN_Force/openmmapi/tests/:/home/fisiksnju/openmmexampleplugin/build/:/usr/local/openmm/lib/plugins"
	
clean:
	rm test_ANN_Force test_ANN_package _ANN.so libANN.so *.o ANN_wrap.cxx ANN.py* 
