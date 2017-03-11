// real3 force1 = make_real3(- FORCE_CONSTANT[0] * (pos1.x - POTENTIAL_CENTER[0]), 0.0, 0.0) ; 
// real3 force2 = make_real3(0.0, 0.0, 0.0);

int num_of_rows, num_of_cols;

// forward propagation
// input layer
for (int ii = 0; ii < NUM_OF_NODES[0]; ii ++) {
    OUTPUT_0[ii] = INPUT_0[ii];
}
// layer 1
num_of_rows = NUM_OF_NODES[1];
num_of_cols = NUM_OF_NODES[0];
for (int ii = 0; ii < num_of_rows; ii ++) {
    INPUT_1[ii] = BIAS_0[ii];
    for (int jj = 0; jj < num_of_cols; jj ++) {
        INPUT_1[ii] += COEFF_0[ii * num_of_cols + jj] * OUTPUT_0[jj];
    }
}
if (LAYER_TYPES[0] == 0) { // linear
    for (int ii = 0; ii < num_of_rows; ii ++) {
        OUTPUT_1[ii] = INPUT_1[ii];
    }
}
else if (LAYER_TYPES[0] == 1) { // tanh
    for (int ii = 0; ii < num_of_rows; ii ++) {
        OUTPUT_1[ii] = tanh(INPUT_1[ii]);
    }
}
// layer 2
num_of_rows = NUM_OF_NODES[2];
num_of_cols = NUM_OF_NODES[1];
for (int ii = 0; ii < num_of_rows; ii ++) {
    INPUT_2[ii] = BIAS_1[ii];
    for (int jj = 0; jj < num_of_cols; jj ++) {
        INPUT_2[ii] += COEFF_1[ii * num_of_cols + jj] * OUTPUT_1[jj];
    }
}
if (LAYER_TYPES[0] == 0) { // linear
    for (int ii = 0; ii < num_of_rows; ii ++) {
        OUTPUT_2[ii] = INPUT_2[ii];
    }
}
else if (LAYER_TYPES[0] == 1) { // tanh
    for (int ii = 0; ii < num_of_rows; ii ++) {
        OUTPUT_2[ii] = tanh(INPUT_2[ii]);
    }
}

// backward propagation, INPUT_{0,1,2} are reused to store derivatives in each layer
// layer 2
for (int ii = 0; ii < NUM_OF_NODES[2]; ii ++) {
    INPUT_2[ii] = (OUTPUT_2[ii] - POTENTIAL_CENTER[ii]) * FORCE_CONSTANT[0];
}
if (LAYER_TYPES[1] == 1) {
    for (int ii = 0; ii < NUM_OF_NODES[2]; ii ++) {
        INPUT_2[ii] *= (1 - OUTPUT_2[ii] * OUTPUT_2[ii]);    
    }
}

// layer 1
num_of_rows = NUM_OF_NODES[2];
num_of_cols = NUM_OF_NODES[1];
for (int ii = 0; ii < num_of_cols; ii ++) {
    INPUT_1[ii] = 0;
    for (int jj = 0; jj < num_of_rows; jj ++) {
        INPUT_1[ii] += COEFF_1[ii + jj * num_of_cols] * INPUT_2[jj];
    }
}
if (LAYER_TYPES[1] == 1) {
    for (int ii = 0; ii < NUM_OF_NODES[1]; ii ++) {
        INPUT_1[ii] *= (1 - OUTPUT_1[ii] * OUTPUT_1[ii]);    
    }
}

// input layer
num_of_rows = NUM_OF_NODES[1];
num_of_cols = NUM_OF_NODES[0];
for (int ii = 0; ii < num_of_cols; ii ++) {
    INPUT_0[ii] = 0;
    for (int jj = 0; jj < num_of_rows; jj ++) {
        INPUT_0[ii] += COEFF_0[ii + jj * num_of_cols] * INPUT_1[jj];
    }
}
if (LAYER_TYPES[1] == 1) {
    for (int ii = 0; ii < NUM_OF_NODES[0]; ii ++) {
        INPUT_0[ii] *= (1 - OUTPUT_0[ii] * OUTPUT_0[ii]);    
    }
}

