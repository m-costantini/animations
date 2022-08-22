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

from math import pi, sin
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# generate parametric rotated ellipse for plotting


# Ellipse
theta = pi/4 # tilting angle = 45 degrees
t = np.linspace(0,2*pi,60) # parameter of the curve
Rx = 1
Ry = 0.5
x_ell = Rx*np.cos(t)*np.cos(theta) - Ry*np.sin(t)*np.sin(theta)
y_ell = Rx*np.cos(t)*np.sin(theta) + Ry*np.sin(t)*np.cos(theta)

# Circle
R = 0.8
x_circ = R*np.cos(t)
y_circ = R*np.sin(t)

titles = ['Circle', 'Ellipse 1', 'Ellipse 2']
fig = make_subplots(rows=1, cols=3, subplot_titles=titles)
fig.add_trace(go.Scatter(x=x_circ, y=y_circ), row=1, col=1)
fig.add_trace(go.Scatter(x=x_ell, y=y_ell), row=1, col=2)
fig.add_trace(go.Scatter(x=x_ell, y=y_ell), row=1, col=3)


fig.update_yaxes(range=[-1,1], scaleanchor="x", scaleratio=1, constrain="domain")
fig.update_xaxes(range=[-1,1])

n_rounds = 3
n_points = 100
delta_radius = 0.03
for r in range(n_rounds):

    Rx += delta_radius
    Ry += delta_radius
    R += delta_radius

    t = np.random.uniform(low=0.0, high=2*pi, size=(n_points,))

    x_ell = Rx*np.cos(t)*np.cos(theta) - Ry*np.sin(t)*np.sin(theta)
    y_ell = Rx*np.cos(t)*np.sin(theta) + Ry*np.sin(t)*np.cos(theta)
    x_circ = R*np.cos(t)
    y_circ = R*np.sin(t)

    fig.add_trace(go.Scatter(x=list(x_circ), y=list(y_circ),
        mode='markers', marker=dict(size=3, color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(x_ell), y=list(y_ell),
        mode='markers', marker=dict(size=3, color='red')), row=1, col=2)
    fig.update_layout(showlegend=False)

    # # save this frame
    # save_name = 'sampling_on_ellipse_figures/ellipse_fig_' + str(r+1) + '.png'
    # fig.write_image(save_name)
    # print('Saved figure', r+1, '/', n_rounds)

fig.show()
