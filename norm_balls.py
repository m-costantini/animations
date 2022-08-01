import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import sqrt, pi

# L2 ball
u, v = np.mgrid[0:2*np.pi:500j, 0:2*np.pi:500j]
x2 = np.sin(u) * np.cos(v)
y2 = np.sin(u) * np.sin(v)
z2 = np.cos(u)
fig = go.Figure(data=[go.Surface(x=x2, y=y2, z=z2, opacity=0.5, colorscale="Bluered", showscale=False)])
fig.show()

# L1 ball by normalization
rho = 1/(np.abs(x2) + np.abs(y2) + np.abs(z2))
x1 = rho * np.sin(u) * np.cos(v)
y1 = rho * np.sin(u) * np.sin(v)
z1 = rho * np.cos(u)
fig = go.Figure(data=[go.Surface(x=x1, y=y1, z=z1, opacity=0.5, colorscale="Bluered", showscale=False)])
fig.show()


# Linf ball by normalization
M = np.concatenate((x2[:,:,np.newaxis], y2[:,:,np.newaxis], z2[:,:,np.newaxis]), axis=2)
rho = 1/np.max(np.abs(M),axis=2)
xinf = rho * np.sin(u) * np.cos(v)
yinf = rho * np.sin(u) * np.sin(v)
zinf = rho * np.cos(u)
fig = go.Figure(data=[go.Surface(x=xinf, y=yinf, z=zinf, opacity=0.5, colorscale="Bluered", showscale=False)])
fig.show()


# Lhalf ball by normalization
rho = 1/( np.sqrt(np.abs(x2)) + np.sqrt(np.abs(y2)) + np.sqrt(np.abs(z2)) )**2
xhalf = rho * np.sin(u) * np.cos(v)
yhalf = rho * np.sin(u) * np.sin(v)
zhalf = rho * np.cos(u)
fig = go.Figure(data=[go.Surface(x=xhalf, y=yhalf, z=zhalf, opacity=0.5, colorscale="Bluered", showscale=False)])
fig.show()
