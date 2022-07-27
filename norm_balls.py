import numpy as np
import plotly.graph_objects as go
from math import sqrt

n_points = 1000
x_support = np.linspace(-1, 1, n_points)
y_support = np.linspace(-1, 1, n_points)
x, y = np.meshgrid(x_support,y_support)

camera = dict( eye=dict(x=2, y=2, z=1) )

sum_quares = lambda x, y: x**2 + y**2

v3 = lambda v1v2: sqrt(1-sum_quares(v1v2[0],v1v2[1])) if sum_quares(v1v2[0],v1v2[1]) <= 1 else None
get_z = lambda v1, v2: np.array(list(map(v3, list(zip(v1,v2)))))

z = get_z(x.flatten(),y.flatten()).reshape(np.shape(x))
z_flat = z.flatten()
z_minus = [None for i in range(len(z_flat))]

# print(z)
# print(z_minus)


for i in range(len(z_flat)):
    if z_flat[i] is not None:
        z_minus[i] = -z_flat[i]

z_minus = np.array(z_minus).reshape(np.shape(z))

fig = go.Figure(data=[go.Surface(z=z, x=x_support, y=y_support, opacity=0.5, colorscale="Bluered", showscale=False)])
fig.add_trace(go.Surface(z=z_minus, x=x_support, y=y_support, opacity=0.5, colorscale="Bluered", showscale=False))

#
# fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True,
#         start=dic['start'], end=dic['end'], size=dic['size'], width=16.0, highlightwidth=10.0, highlight=True))

aux = 5
margin=go.layout.Margin(l=aux, r=aux, b=aux, t=aux, pad = 0)
fig.update_layout(scene_camera=camera, autosize=False, width=800, height=800, margin=margin)
#
# fontsize = 50
# fig.update_layout(scene = dict(xaxis_title=dict(text="λ<sub>1</sub>", font_size=fontsize),
#                                yaxis_title=dict(text="λ<sub>2</sub>", font_size=fontsize),
#                                zaxis_title='', zaxis_color='#ffffff'))


fig.show()
