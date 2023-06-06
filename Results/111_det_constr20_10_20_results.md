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

 total control cost: 1415.4733276586408

 time to touchdown: 12.65 s

 touchdown velocities: [-0.13916803 -0.03883588  0.01056713] m/s, m/s, rad/s

 time to simulate: 244.8 s

 Parameters: N_scp: 10 / N_mpc: 20 / known_pad_dynamics: True / noise variance: 0.0 / wind: False / rs: inf / ru: 20.0 / rT: inf / rdu: 5.0