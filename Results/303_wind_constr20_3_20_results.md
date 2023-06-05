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

 total control cost: 16497.96647271686

 time to touchdown: 0 s

 touchdown velocities: [0. 0. 0.] m/s, m/s, rad/s

 time to simulate: 217.1 s

 Parameters: N_scp: 3 / N_mpc: 20 / known_pad_dynamics: True / noise variance: 0.0 / wind: True / rs: inf / ru: 20.0 / rT: inf / rdu: 5.0