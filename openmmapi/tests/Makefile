OpenMM_INSTALL_DIR=/usr/local/openmm
CFLAGS = -g -std=c++11


FCPPLIBS = -lstdc++

LIB_DIR=$(OpenMM_INSTALL_DIR)/lib
INCLUDE_DIR=$(OpenMM_INSTALL_DIR)/include
LIBS= -lOpenMM

test:
	g++ $(CFLAGS) -I$(INCLUDE_DIR) test_ANN_Force.cpp  -L$(LIB_DIR) $(LIBS) -o test

clean:
	rm test
