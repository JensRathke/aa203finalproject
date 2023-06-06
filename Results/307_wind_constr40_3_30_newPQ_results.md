Controller: constraint non-linear MPC

 P / QN:
 [[ 500.    0.    0.    0.    0.    0.]
 [   0.  500.    0.    0.    0.    0.]
 [   0.    0.  100.    0.    0.    0.]
 [   0.    0.    0.  200.    0.    0.]
 [   0.    0.    0.    0. 1000.    0.]
 [   0.    0.    0.    0.    0.  100.]]

 Q:
 [[ 5.  0.  0.  0.  0.  0.]
 [ 0.  5.  0.  0.  0.  0.]
 [ 0.  0.  2.  0.  0.  0.]
 [ 0.  0.  0. 10.  0.  0.]
 [ 0.  0.  0.  0. 40.  0.]
 [ 0.  0.  0.  0.  0. 10.]]

 R:
 [[0.1 0. ]
 [0.  0.1]]

 total control cost: 1407.776202366309

 time to touchdown: 10.850000000000001 s

 touchdown velocities: [0.71243221 0.04747336 0.01094813] m/s, m/s, rad/s

 time to simulate: 222.51 s

 Parameters: N_scp: 3 / N_mpc: 30 / known_pad_dynamics: True / noise variance: 0.0 / wind: True / rs: inf / ru: 40.0 / rT: inf / rdu: 5.0