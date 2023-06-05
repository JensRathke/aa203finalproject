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

 total control cost: 12855.360096138473

 time to touchdown: 12.9 s

 touchdown velocities: [-0.21104526 -0.01178564  0.02911751] m/s, m/s, rad/s

 time to simulate: 160.75 s

 Parameters: N_scp: 3 / N_mpc: 20 / known_pad_dynamics: True / noise variance: 0.0 / wind: True / rs: inf / ru: 40.0 / rT: inf / rdu: 5.0