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

 total control cost: 2284.4463212535893

 time to touchdown: 19.200000000000003 s

 touchdown velocities: [1.32119051 0.03902774 0.00459448] m/s, m/s, rad/s

 time to simulate: 410.21 s

 Parameters: N_scp: 3 / N_mpc: 30 / known_pad_dynamics: True / noise variance: 0.0 / wind: True / rs: inf / ru: 40.0 / rT: inf / rdu: 5.0