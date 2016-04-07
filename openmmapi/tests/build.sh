swig -c++ -python ANN.i
g++ -std=c++11 -I/home/fisiksnju/.anaconda2/include/python2.7/ -I/usr/local/openmm/include \
-I/home/fisiksnju/Dropbox/temp_Linux/ANN_Force/openmmapi/include -I/usr/local/openmm/include/openmm/reference \
-shared -fPIC -c ANN_wrap.cxx test_ANN_package.cpp

g++ -shared -I/home/fisiksnju/.anaconda2/include/python2.7/ -I/usr/local/openmm/include \
-I/home/fisiksnju/Dropbox/temp_Linux/ANN_Force/openmmapi/include -I/usr/local/openmm/include/openmm/reference *.o -o _ANN.so \
-L/usr/local/openmm/lib -lOpenMM -L/home/fisiksnju/Dropbox/temp_Linux/ANN_Force/openmmapi/tests/ -lANN
