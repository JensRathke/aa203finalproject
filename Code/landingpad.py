import numpy as np
import matplotlib.pyplot as plt

class landingpad:

  def __init__(self, t = np.arange(0,0.1,100), x_lim = 100, y_lim = 100, z_lim = 100, dz_max = 5, pad_len = 10, dtheta_max = np.pi/180.*10, num_samples = 1000, N_max = 20000):

    mu, sigma = 0., 1.
    x = self.get_brownian( mu, sigma, N_max, num_samples)
    y = self.get_brownian( mu, sigma, N_max, num_samples)
    z = self.get_brownian( mu, sigma, N_max, num_samples)
    theta = self.get_brownian( mu, sigma, N_max, num_samples)
    psi = self.get_brownian( mu, sigma, N_max, num_samples)

    tau = t

    self.x_lim = x_lim #max size for x dimension
    self.y_lim = y_lim

    self.pad_len = 10
    f_coef = [1.,1.,1., 1., 1., 1., 1., 1., 1., 1.]
    x = self.fir_filter(x,f_coef)
    y = self.fir_filter(y,f_coef)
    z = self.fir_filter(z,f_coef)
    theta = self.fir_filter(theta,f_coef)
    psi = self.fir_filter(psi,f_coef)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    theta_mean = np.mean(theta)
    psi_mean = np.mean(psi)
    x = x - x_mean
    y = y - y_mean
    z = z - z_mean
    theta = theta - theta_mean
    psi = psi - psi_mean
    x_min = np.min(x)
    y_min = np.min(y)
    z_min = np.min(z)
    theta_min = np.min(theta)
    psi_min = np.min(psi)
    x_max = np.max(x)
    y_max = np.max(y)
    z_max = np.max(z)
    psi_max = np.max(psi)
    theta_max = np.max(theta)
    x_range= np.max([np.abs(x_max),np.abs(x_min)])
    y_range= np.max([np.abs(y_max),np.abs(y_min)])
    z_range= np.max([np.abs(z_max),np.abs(z_min)])
    theta_range= np.max([np.abs(theta_max),np.abs(theta_min)])
    psi_range= np.max([np.abs(psi_max),np.abs(psi_min)])
    x /=x_range
    y /=y_range
    z /=z_range
    psi /=psi_range
    theta /=theta_range
    x *= (x_lim/2)
    y *= (y_lim/2)
    z *= (dz_max /2)
    psi *= (dtheta_max/2)
    theta *= (dtheta_max/2)
    x += (x_lim/2)
    y += (y_lim/2)

    dz_x = 2.5 *np.cos(2 * np.pi*tau + theta)
    dz_y = 2.5 * np.cos(2 * np.pi*tau + psi)

    self.x = x
    self.y = y
    self.theta = theta
    self.psi = psi
    self.z = z

  def get_brownian(self, mu, sigma, N, target_num):
    """
    simulate brownian motion as cumulative sum of (N+target_num) N(mu,sigma)
    downsample to target_num
    """
    M = N+target_num
    x = np.random.normal(mu,sigma,M)
    x = np.cumsum(x)
    stride = N/target_num
    stride= np.rint(stride).astype(int)
    print('stride',stride)
    y = x[target_num-1:-1:stride]
    #y = x[0:-1:stride]
    return y


  #for smoothing - can be replaced with 1 pole iir
  def fir_filter(self, x, f):
    return np.convolve(x, f, 'same') / np.size(f)

  def updateposition2d(self):
    #return self.x_pos, self.y_pos
    return self.x, self.y, self.z, self.theta, self.psi
  def updateposition1d(self):
    #return self.x_pos, self.y_pos
    return self.x,  self.z, self.theta

if __name__ == "__main__":
  pass
"""
  l_pad =  landingpad()
  #for i in range(100):
   # print ('x', l_pad.updateposition()[0], 'y', l_pad.updateposition()[1])

x_pos = l_pad.updateposition2d()[0]
y_pos = l_pad.updateposition2d()[1]
print ('x', x_pos[-10:-1], 'y', y_pos[-10:-1])
fig, ax = plt.subplots(dpi=100)
ax.set_xlim([0., 100.])
ax.set_ylim([0., 100.])
#ax.plot(x_pos, y_pos, '--', linewidth=5, color='tab:orange')[0]
ax.plot(x_pos, y_pos, linewidth=2, color='tab:orange')
plt.show()
"""




