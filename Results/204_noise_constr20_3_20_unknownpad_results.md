Controller: constraint non-linear MPC

 P / QN:
 [[ 500.    0.    0.    0.    0.    0.]
 [   0.  500.    0.    0.    0.    0.]
 [   0.    0.  100.    0.    0.    0.]
 [   0.    0.    0.  500.    0.    0.]
 [   0.    0.    0.    0. 1000.    0.]
 [   0.    0.    0.    0.    0.  100.]]

 Q:
 [[ 5.  0.  0.  0.  0.  0.]
 [ 0.  5.  0.  0.  0.  0.]
 [ 0.  0.  2.  0.  0.  0.]
 [ 0.  0.  0. 30.  0.  0.]
 [ 0.  0.  0.  0. 40.  0.]
 [ 0.  0.  0.  0.  0. 10.]]

 R:
 [[0.1 0. ]
 [0.  0.1]]

 total control cost: 1449.2869143846674

 time to touchdown: 12.850000000000001 s

 touchdown velocities: [-0.2442367  -0.03693419  0.02171565] m/s, m/s, rad/s

 time to simulate: 143.51 s

 Parameters: N_scp: 3 / N_mpc: 20 / known_pad_dynamics: False / noise variance: 0.4 / wind: False / rs: inf / ru: 20.0 / rT: inf / rdu: 5.0