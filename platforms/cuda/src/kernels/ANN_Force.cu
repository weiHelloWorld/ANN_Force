float2 bondParams = PARAMS[0];
float temp = (float)PARAMS_1[0];
real3 force1 = make_real3(- bondParams.y * (pos1.x - bondParams.x), - bondParams.y * (pos1.y - temp), 0.0) ; 
real3 force2 = make_real3(0.0, 0.0, 0.0);
