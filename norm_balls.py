import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import sqrt, pi

# # L2 ball
u, v = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]
x = np.sin(u) * np.cos(v)
y = np.sin(u) * np.sin(v)
z = np.cos(u)
fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, opacity=0.5, colorscale="Bluered", showscale=False)])
fig.show()


# # L1 ball - plot all 8 surfaces
# do grid by hand to get triangular face
n_points = 2
u_1D = np.linspace(0,0.5,n_points)
u = np.zeros((n_points,n_points))
t = np.zeros((n_points,n_points))
for idx, u_val in enumerate(u_1D):
    t_1D = np.linspace(u_val,1-u_val,n_points)
    u[idx,:] = u_val*np.ones((n_points,))
    t[idx,:] = t_1D
x_plus = t - u
x_minus = -(t - u)
y_plus = 1 - t - u
y_minus = -(1 - t - u)
z_plus = 2*u
z_minus = -2*u
fig = go.Figure(data=[go.Surface(x=x_plus, y=y_plus, z=z_plus, opacity=0.5, colorscale="Bluered", showscale=False)])
fig.add_trace(go.Surface(x=x_plus, y=y_minus, z=z_plus, opacity=0.5, colorscale="Bluered", showscale=False))
fig.add_trace(go.Surface(x=x_minus, y=y_minus, z=z_plus, opacity=0.5, colorscale="Bluered", showscale=False))
fig.add_trace(go.Surface(x=x_minus, y=y_plus, z=z_plus, opacity=0.5, colorscale="Bluered", showscale=False))
fig.add_trace(go.Surface(x=x_plus, y=y_plus, z=z_minus, opacity=0.5, colorscale="Bluered", showscale=False))
fig.add_trace(go.Surface(x=x_plus, y=y_minus, z=z_minus, opacity=0.5, colorscale="Bluered", showscale=False))
fig.add_trace(go.Surface(x=x_minus, y=y_minus, z=z_minus, opacity=0.5, colorscale="Bluered", showscale=False))
fig.add_trace(go.Surface(x=x_minus, y=y_plus, z=z_minus, opacity=0.5, colorscale="Bluered", showscale=False))
fig.show()


# Infinite ball
linear_face_1, linear_face_2 = np.mgrid[-1:1:2j, -1:1:2j]
constant_face = np.ones(np.shape(linear_face_1))
fig = go.Figure(data=[go.Surface(x=constant_face, y=linear_face_1, z=linear_face_2, opacity=0.5, colorscale="Bluered", showscale=False)])
fig.add_trace(go.Surface(x=-constant_face, y=linear_face_1, z=linear_face_2, opacity=0.5, colorscale="Bluered", showscale=False))
fig.add_trace(go.Surface(x=linear_face_2, y=constant_face, z=linear_face_1, opacity=0.5, colorscale="Bluered", showscale=False))
fig.add_trace(go.Surface(x=linear_face_2, y=-constant_face, z=linear_face_1, opacity=0.5, colorscale="Bluered", showscale=False))
fig.add_trace(go.Surface(x=linear_face_1, y=linear_face_2, z=constant_face, opacity=0.5, colorscale="Bluered", showscale=False))
fig.add_trace(go.Surface(x=linear_face_1, y=linear_face_2, z=-constant_face, opacity=0.5, colorscale="Bluered", showscale=False))
fig.show()
