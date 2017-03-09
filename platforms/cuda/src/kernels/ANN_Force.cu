float temp = (float) PARAMS_1[0];
float force_constant = (float) FORCE_CONSTANT[0];
real3 force1 = make_real3(- force_constant * (pos1.x - POTENTIAL_CENTER[0]), - force_constant * (pos1.y - temp), 0.0) ; 
real3 force2 = make_real3(0.0, 0.0, 0.0);
