"""
Code to sample uniformly from an ellipse

Follows the explanation of Sebastien Bubeck in his lecture "Fice miracles of Mirror Descent"
https://youtu.be/36zUR3QdOD0?list=PLAPSKVSdi0obG1b3w4k41JMLFbyBJS5AQ&t=2280
(Lecture 8/9, minute 38:00)

References used for this code:


https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotatio

August 2022
marina.costant@gmail.com
"""

from math import pi, sqrt
import numpy as np
from scipy import linalg

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt


# generate parametric rotated ellipse for plotting


# Ellipse
theta = pi/4 # tilting angle = 45 degrees
t = np.linspace(0,2*pi,60) # parameter of the curve
Rx = 1
Ry = 0.5
x_ell = Rx*np.cos(t)*np.cos(theta) - Ry*np.sin(t)*np.sin(theta)
y_ell = Rx*np.cos(t)*np.sin(theta) + Ry*np.sin(t)*np.cos(theta)

# Circle
R = 1
x_circ = R*np.cos(t)
y_circ = R*np.sin(t)

# Ellipse from circle
a = sqrt(Rx)
b = sqrt(Ry)
A = a**2*np.sin(theta)**2 + b**2*np.cos(theta)**2
B = 2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
C = a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2
V = np.vstack((x_circ,y_circ))
S = np.array([[A,B/2],[B/2,C]])
W = S @ V
# print(S)
# S_inv = linalg.inv(S)
# print(S_inv)
S_inv_sqrt = linalg.inv(linalg.sqrtm(S))
# print(S_inv_sqrt)
x_ell_2 = W[0,:]
y_ell_2 = W[1,:]



# -----------------------------------------------------
# Plots ellipses with plotly
# -----------------------------------------------------
# Figure
# titles = ['Circle', 'Ellipse 1', 'Ellipse 2']
# subplot_specs = {'type': 'barpolar'}
# fig = make_subplots(rows=1, cols=3, subplot_titles=titles, specs=[[subplot_specs, subplot_specs, subplot_specs]])
#
#
# # fig.add_trace(go.Scatter(x=0.8*x_circ, y=0.8*y_circ), row=1, col=1) # 0.8 for it to look better
# # fig.add_trace(go.Scatter(x=x_ell, y=y_ell), row=1, col=2)
# # fig.add_trace(go.Scatter(x=x_ell_2, y=y_ell_2), row=1, col=3)
#
# fig.update_yaxes(range=[-2,2], scaleanchor="x", scaleratio=1, constrain="domain")
# fig.update_xaxes(range=[-2,2])

# fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")

# -----------------------------------------------------
# Plotting points
# -----------------------------------------------------
# n_rounds = 20
# n_points = 50
# delta_radius = 0.03
# for r in range(n_rounds):
#
#     # Rx += delta_radius
#     # Ry += delta_radius
#     # R += delta_radius
#
#     Rx *= 1.02
#     Ry *= 1.02
#     R *= 1.02
#
#     t = np.random.uniform(low=0.0, high=2*pi, size=(n_points,))
#
#     x_ell = Rx*np.cos(t)*np.cos(theta) - Ry*np.sin(t)*np.sin(theta)
#     y_ell = Rx*np.cos(t)*np.sin(theta) + Ry*np.sin(t)*np.cos(theta)
#     x_circ = R*np.cos(t)
#     y_circ = R*np.sin(t)
#     V = np.vstack((x_circ,y_circ))
#     W = S_inv_sqrt @ V
#     x_ell_2 = W[0,:]
#     y_ell_2 = W[1,:]
#
#     fig.add_trace(go.Scatter(x=list(x_circ), y=list(y_circ),
#         mode='markers', marker=dict(size=3, color='blue')), row=1, col=1)
#     fig.add_trace(go.Scatter(x=list(x_ell), y=list(y_ell),
#         mode='markers', marker=dict(size=3, color='red')), row=1, col=2)
#     fig.add_trace(go.Scatter(x=list(x_ell_2), y=list(y_ell_2),
#         mode='markers', marker=dict(size=3, color='green')), row=1, col=3)
#
#     fig.update_layout(showlegend=False)

    # # save this frame
    # save_name = 'sampling_on_ellipse_figures/ellipse_fig_' + str(r+1) + '.png'
    # fig.write_image(save_name)
    # print('Saved figure', r+1, '/', n_rounds)


# -----------------------------------------------------
# Polar plot in plotly
# -----------------------------------------------------
# n_points = 10000
#
# t = np.random.uniform(low=0.0, high=2*pi, size=(n_points,))
#
# # x_ell = Rx*np.cos(t)*np.cos(theta) - Ry*np.sin(t)*np.sin(theta)
# # y_ell = Rx*np.cos(t)*np.sin(theta) + Ry*np.sin(t)*np.cos(theta)
# x_circ = R*np.cos(t)
# y_circ = R*np.sin(t)
# V = np.vstack((x_circ,y_circ))
# W = S_inv_sqrt @ V
# x_ell_2 = W[0,:]
# y_ell_2 = W[1,:]
#
# t_prime = np.arctan2(y_ell_2, x_ell_2) + pi # +pi because arctan2 returns values in -pi:pi
#
# n_bins = 36
# bins_pi = np.linspace(0,2*pi,n_bins+1)
# bins = bins_pi*360/(2*pi)
# d_bin = bins[1]
#
# counts = np.histogram(t, bins=bins_pi)[0]
# norm_counts_t = counts/max(counts)
# counts = np.histogram(t_prime, bins=bins_pi)[0]
# norm_counts_t_prime = counts/max(counts)
#
# fig.add_trace(go.Barpolar(
#     r=norm_counts_t,
#     theta=bins+(d_bin/2),
#     width=d_bin*np.ones((len(norm_counts_t),)),
#     marker_line_color="black",
#     marker_line_width=2,
#     opacity=0.8
#     ), row=1, col=1)
#
# fig.add_trace(go.Barpolar(
#     r=norm_counts_t_prime,
#     theta=bins+(d_bin/2),
#     width=d_bin*np.ones((len(norm_counts_t_prime),)),
#     marker_line_color="black",
#     marker_line_width=2,
#     opacity=0.8
#     ), row=1, col=2)
#
# fig.update_layout(showlegend=False)
# fig.show()







# -----------------------------------------------------
# Polar plot in matplotlib
# -----------------------------------------------------


fig, ax = plt.subplots(1,2, figsize=(15,4), subplot_kw={'projection': 'polar'}, dpi=100)


n_points = 10000
t = np.random.uniform(low=0.0, high=2*pi, size=(n_points,))
x_circ = np.cos(t)
y_circ = np.sin(t)
V = np.vstack((x_circ,y_circ))
W = S_inv_sqrt @ V
x_ell = W[0,:]
y_ell = W[1,:]
t_prime = np.arctan2(y_ell_2, x_ell_2) + pi # +pi because arctan2 returns values in -pi:pi

n_bins = 36
bins_pi = np.linspace(0,2*pi,n_bins+1)
bins = bins_pi*360/(2*pi)
d_bin = bins_pi[1]


counts = np.histogram(t, bins=bins_pi)[0]
norm_counts_t = counts/max(counts)
counts = np.histogram(t_prime, bins=bins_pi)[0]
norm_counts_t_prime = counts/max(counts)

print(bins[:-1], counts, d_bin)

#
# ax[0].bar(
#     x=[0,np.pi/4,np.pi/2],
#     height=[1,2,3],
#     bottom=0,
#     width=np.pi/16)


ax[0].bar(
    x=bins_pi[:-1]+d_bin/2,
    height=norm_counts_t,
    bottom=0,
    width=0.9*d_bin)


plt.show()
