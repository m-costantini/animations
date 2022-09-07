"""
Code to sample uniformly from the boundary of an ellipse

Follows the explanation of Sebastien Bubeck in his lecture "Five miracles of Mirror Descent"
https://youtu.be/36zUR3QdOD0?list=PLAPSKVSdi0obG1b3w4k41JMLFbyBJS5AQ&t=2280
(Lecture 8/9, minute 38:00)

References used for this code:
https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotation
https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
https://www.geeksforgeeks.org/using-matplotlib-for-animations/
https://en.wikipedia.org/wiki/N-sphere

August 2022
marina.costant@gmail.com
"""

from math import pi, sqrt
import numpy as np
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["mathtext.fontset"] = 'cm' # for Latex-like typesetting the titles

# Circle and ellipse settings
theta = pi/4 # tilting angle = 45 degrees
R = 1 # circle radius
Rx = 1.4 # ellipse large axis
Ry = 0.7 # ellipse small axis

# S and Sinv
a = sqrt(Rx)
b = sqrt(Ry)
A = a**2*np.sin(theta)**2 + b**2*np.cos(theta)**2
B = 2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
C = a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2
S = np.array([[A,B/2],[B/2,C]])
S_inv_sqrt = linalg.inv(linalg.sqrtm(S))

def get_points_ellipse(points_circle, S_inv_sqrt):
    """
    Transformation f: u --> S^{-1/2}u so that if u is uniformly distributed in the boundary of the circle, then f(u) is uniformly distributed on the boundary of the ellipse
    """
    x_circle = np.cos(points_circle)
    y_circle = np.sin(points_circle)
    V = np.vstack((x_circle, y_circle))
    W = S_inv_sqrt @ V
    x_ell_Sis = W[0,:] # Sis = S inverse square
    y_ell_Sis = W[1,:]
    points_ellipse = np.arctan2(x_ell_Sis, y_ell_Sis) + pi # +pi because arctan2 gives results in [-pi, +pi]
    return points_ellipse

fig, ax = plt.subplots(1,2, figsize=(8,4), subplot_kw={'projection': 'polar'}, dpi=100)

# Settings for plots
points_scatter = 70
points_bars = 10000
n_frames = 10
n_bins = 36
bins_pi = np.linspace(0,2*pi,n_bins+1)
bins_degrees = bins_pi*360/(2*pi)
delta_bin = bins_pi[1]

def animate(f):

    ax[0].clear()
    ax[1].clear()

    # Scatter plots
    points_circle = np.random.uniform(low=0.0, high=2*pi, size=(points_scatter,))
    points_ellipse = get_points_ellipse(points_circle, S_inv_sqrt)
    ax[0].scatter(points_circle, 1.2*np.ones((len(points_circle),)), c='cornflowerblue')
    ax[1].scatter(points_ellipse, 1.6*np.ones((len(points_ellipse),)), c='indianred')
    # uncomment  the two lines below for increasing radius in the scatter plot
    # ax[0].scatter(points_circle, (1+0.1*f)*np.ones((len(points_circle),)), s=5, c='mediumblue')
    # ax[1].scatter(points_ellipse, (1+0.1*f)*np.ones((len(points_ellipse),)), s=5, c='darkred')

    # Bar plots
    points_circle = np.random.uniform(low=0.0, high=2*pi, size=(int((f+1)*points_bars/n_frames),))
    points_ellipse = get_points_ellipse(points_circle, S_inv_sqrt)
    counts_circle = np.histogram(points_circle, bins=bins_pi)[0]
    norm_counts_circle = n_bins*counts_circle/points_bars
    counts_ellipse = np.histogram(points_ellipse, bins=bins_pi)[0]
    norm_counts_ellipse  = n_bins*counts_ellipse/points_bars
    ax[0].bar(x=bins_pi[:-1]+delta_bin/2, height=norm_counts_circle, color='cornflowerblue', width=0.9*delta_bin)
    ax[1].bar(x=bins_pi[:-1]+delta_bin/2, height=norm_counts_ellipse, color='indianred', width=0.9*delta_bin)

    # Circle / ellipse
    t = np.linspace(0,2*pi,60) # parameter of the curve
    x_circle = R*np.cos(t)
    y_circle = R*np.sin(t)
    x_ellipse = Rx*np.cos(t)*np.cos(theta) - Ry*np.sin(t)*np.sin(theta)
    y_ellipse = Rx*np.cos(t)*np.sin(theta) + Ry*np.sin(t)*np.cos(theta)
    theta_circle = np.linspace(0,2*pi,36)
    radius_circle = R*np.ones((36,))
    theta_ellipse = np.arctan2(y_ellipse, x_ellipse) + pi
    radius_ellipse =  np.sqrt(y_ellipse**2 + x_ellipse**2)
    ax[0].plot(theta_circle, radius_circle, color='mediumblue', linewidth=2)
    ax[1].plot(theta_ellipse, radius_ellipse, color='darkred', linewidth=2)

    # Plot formatting
    # ax[0].set_ylim((0,2))
    # ax[1].set_ylim((0,2))
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[1].set_yticklabels([])
    ax[0].set_title(r'$\mathrm{\mathbf{Circle}}$'
                    '\n'
                    r'$\mathrm{Equation{:}} x^TIx=1$'
                    '\n'
                    r'$\mathrm{Sampling{:}} s \sim \mathcal{U}(S^1)$', fontsize=18, pad=10)
    ax[1].set_title(r'$\mathrm{\mathbf{Ellipse}}$'
                    '\n'
                    r'$\mathrm{Equation{:}} x^T \Sigma x=1$'
                    '\n'
                    r'$\mathrm{Sampling{:}} s \sim \Sigma^{-1/2}\mathcal{U}(S^1)$', fontsize=18, pad=10)
    fig.subplots_adjust(
        left  = 0.125,  # the left side of the subplots of the figure
        right = 0.9,    # the right side of the subplots of the figure
        bottom = 0.05,   # the bottom of the subplots of the figure
        top = 0.71,      # the top of the subplots of the figure
        wspace = 0.25,   # the amount of width reserved for blank space between subplots
        hspace = 0.2)   # the amount of height reserved for white space between subplots
    return ax

anim = FuncAnimation(fig, animate, frames=10, interval=10000, repeat=True)
anim.save('gifs/sampling_on_ellipse.gif', writer='pillow', fps=3) # changed by MC
# plt.savefig('last_frame.png', dpi=None) # uncomment to save the last frame as png
