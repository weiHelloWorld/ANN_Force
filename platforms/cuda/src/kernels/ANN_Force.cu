float2 bondParams = PARAMS[0];
real3 force1 = make_real3(- bondParams.y * (pos1.x - bondParams.x), 0.0, 0.0) ; 
real3 force2 = make_real3(0.0, 0.0, 0.0);
