#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation

filepath = "Animations/"

def animate_planar_quad(filename, t, x, y, phi, x_goal, y_goal, phi_goal, l, r, h):
    """
    Based on: Spencer M. Richards
              Autonomous Systems Lab (ASL), Stanford
              (GitHub: spenrich)
    
    Functionality
        Animations of a planar quadcopter
    
    Parameters
        filename: name and relative directory of the output file without file extention (e.g.: "dir/filename")
        t: array of size N with the sequence of time stamps
        x: array of size N with the x coordinates
        y: array of size N with the y coordinates
        phi: array of size N with the angle of the quadcopter
        x_goal: array of size N with the x coordinates of the goal state (landing pad)
        y_goal: array of size N with the y coordinates (landing pad)
        phi_goal: array of size N with the angle of the quadcopter (landing pad)
        l: length of a rotor arm
        r: radius of the cabin
        h: height of the center of mass above skid surface
    """
    # Geometry
    rod_width = 2 * l
    rod_height = 0.2
    axle_height = 0.3
    axle_width = 0.2
    prop_width = 0.8 * rod_width
    prop_height = 0.22
    hub_width = 2 * r
    hub_height = 2 * r
    skid_width = 0.3
    skid_height = 0.3
    com_height = h
    rod_ratio = 0.6

    # Figure and axis
    fig, ax = plt.subplots(dpi=100)
    x_min, x_max = np.min(x), np.max(x)
    x_pad = 1.6 * (rod_width + prop_width)/2 + 0.1*(x_max - x_min)
    y_min, y_max = np.min(y), np.max(y)
    y_pad = (rod_width + prop_width)/2 + 0.1*(y_max - y_min)
    ax.set_xlim([x_min - x_pad, x_max + x_pad])
    ax.set_ylim([y_min - y_pad, y_max + y_pad])
    ax.set_aspect(1.)

    # Artists of the quadcopter
    rod = mpatches.Rectangle((-rod_width/2, rod_ratio * r),
                             rod_width, rod_height,
                             facecolor='tab:blue', edgecolor='k')
    hub = mpatches.FancyBboxPatch((-hub_width/2, -hub_height/2),
                                  hub_width, hub_height,
                                  facecolor='tab:blue', edgecolor='k',
                                  boxstyle='Round,pad=0.,rounding_size=0.5')
    axle_left = mpatches.Rectangle((-rod_width/2, rod_ratio * r + rod_height),
                                   axle_width, axle_height,
                                   facecolor='tab:blue', edgecolor='k')
    axle_right = mpatches.Rectangle((rod_width/2 - axle_width, rod_ratio * r + rod_height),
                                    axle_width, axle_height,
                                    facecolor='tab:blue', edgecolor='k')
    prop_left = mpatches.Ellipse(((axle_width - rod_width)/2,
                                  rod_ratio * r + rod_height + axle_height),
                                 prop_width, prop_height,
                                 facecolor='tab:gray', edgecolor='k',
                                 alpha=0.7)
    prop_right = mpatches.Ellipse(((rod_width - axle_width)/2,
                                   rod_ratio * r + rod_height + axle_height),
                                  prop_width, prop_height,
                                  facecolor='tab:gray', edgecolor='k',
                                  alpha=0.7)
    skid_left = mpatches.Ellipse((-r,
                                  -com_height + skid_height / 2),
                                 skid_width, skid_height,
                                 facecolor='tab:gray', edgecolor='k')
    skid_right = mpatches.Ellipse((r,
                                  -com_height + skid_height / 2),
                                 skid_width, skid_height,
                                 facecolor='tab:gray', edgecolor='k')
    patches = (rod, hub, axle_left, axle_right, prop_left, prop_right, skid_left, skid_right)
    for patch in patches:
        ax.add_patch(patch)
    trace = ax.plot([], [], '--', linewidth=2, color='tab:orange')[0]
    timestamp = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    # Artists of the landing pad
    pad = mpatches.Rectangle((-1.5 * l, -0.1),
                             3 * l, 0.1,
                             facecolor='tab:red', edgecolor='k')
    pad_centre = mpatches.Ellipse((0,
                                   0),
                                  0.4, 0.4,
                                  facecolor='tab:red', edgecolor='k')
    patches_pad = (pad, pad_centre)
    for patch in patches_pad:
        ax.add_patch(patch)

    def animate(k, t, x, y, psi, x_goal, y_goal, psi_goal):
        # Animation of the quadcopter trajectory
        transform = mtransforms.Affine2D().rotate_around(0., 0., psi[k])
        transform += mtransforms.Affine2D().translate(x[k], y[k])
        transform += ax.transData
        for patch in patches:
            patch.set_transform(transform)
        trace.set_data(x[:k+1], y[:k+1])
        timestamp.set_text('t = {:.1f} s'.format(t[k]))

        # Animation of the landing pad
        transform = mtransforms.Affine2D().rotate_around(0., 0., psi_goal[k])
        transform += mtransforms.Affine2D().translate(x_goal[k], y_goal[k])
        transform += ax.transData
        for patch in patches_pad:
            patch.set_transform(transform)

        artists = patches + (trace, timestamp) + patches_pad
        return artists

    dt = t[1] - t[0]
    ani = animation.FuncAnimation(fig, animate, t.size, fargs=(t, x, y, phi, x_goal, y_goal, phi_goal),
                                  interval=dt*1000, blit=True)
    
    ani.save(filepath + filename + '.mp4', writer='ffmpeg')
    print("Saved figure as:", filepath + filename + '.mp4')
    plt.show()


if __name__ == "__main__":
    pass