#! python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation
from landingpad import *


def animate_2d_landing_pad(filename,t, x_goal, y_goal, psi_goal, l, r, h):
    """
    Animations for various dynamical systems using `matplotlib`.

    Author: Spencer M. Richards
            Autonomous Systems Lab (ASL), Stanford
            (GitHub: spenrich)
    """
    """Animate the planar quadrotor system from given position data.

    All arguments are assumed to be 1-D NumPy arrays, where `x`, `y`, and `psi`
    are the degrees of freedom of the planar quadrotor over time `t`.
    """



    # Figure and axis
    fig, ax = plt.subplots(dpi=100)
    x_min, x_max = np.min(x_goal), np.max(x_goal)
    y_min, y_max = np.min(y_goal), np.max(y_goal)
    ax.set_xlim([-20, 120])
    ax.set_ylim([-20, 120])
    ax.set_aspect(1.)

    trace = ax.plot([], [], '--', linewidth=2, color='tab:orange')[0]
    timestamp = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    pad = mpatches.Rectangle((-5, -0.1),
                             10, 0.1,
                             facecolor='tab:red', edgecolor='k')
    pad_centre = mpatches.Ellipse((0,
                                   0),
                                  0.4, 0.4,
                                  facecolor='tab:red', edgecolor='k')
    patches_pad = (pad, pad_centre)
    for patch in patches_pad:
        ax.add_patch(patch)

    def animate(k, t, x_goal, y_goal, psi_goal):

        # Animation of the landing pad
        transform = mtransforms.Affine2D().rotate_around(0., 0., psi_goal[k])
        transform += mtransforms.Affine2D().translate(x_goal[k], y_goal[k])
        transform += ax.transData
        for patch in patches_pad:
            patch.set_transform(transform)

        artists = (trace, timestamp) + patches_pad
        return artists

    dt = t[1] - t[0]
    ani = animation.FuncAnimation(fig, animate, t.size, fargs=(t,  x_goal, y_goal, psi_goal),
                                  interval=dt*1000, blit=True)
   # return fig, ani
    ani.save(filename + '.mp4', writer='ffmpeg')
    plt.show()

if __name__ == "__main__":
  dt = 0.1
  t = np.arange(0,100,dt)
  landing_pad =  landingpad(t = t,num_samples=t.size)
  x,y,theta = landing_pad.updateposition1d()
  #z = np.zeros((t.size,1))
  #print('x', x[:100])
  #print('z', y[:100])
  #print('theta',theta[:100])
  animate_2d_landing_pad("Animations/landing_pad",t, x, y, theta, 10, 10, 10)