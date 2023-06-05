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

 total control cost: 9037.097335656772

 time to touchdown: 12.65 s

 touchdown velocities: [-0.13918841 -0.03894244  0.01057415] m/s, m/s, rad/s

 time to simulate: 158.33 s

 Parameters: N_scp: 3 / N_mpc: 20 / known_pad_dynamics: True / noise variance: 0.0 / wind: False / rs: inf / ru: 20.0 / rT: inf / rdu: 5.0