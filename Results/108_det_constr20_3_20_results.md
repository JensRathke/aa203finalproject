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

 total control cost: 9039.364760750263

 time to touchdown: 12.65 s

 touchdown velocities: [-0.13919954 -0.03901169  0.01057678] m/s, m/s, rad/s

 time to simulate: 145.17 s

 Parameters: N_scp: 3 / N_mpc: 20 / known_pad_dynamics: True / noise variance: 0.0 / wind: False / rs: inf / ru: 20.0 / rT: inf / rdu: 5.0