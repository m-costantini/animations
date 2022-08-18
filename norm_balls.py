"""
Code to create p-unit balls with p = 0.5, 1, 2, infinity

August 2022
marina.costant@gmail.com
"""

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import sqrt, pi

t_vec = np.linspace(0,pi/2,8) # for parametric curve defining camera position

for idx_t, t in enumerate(t_vec[:-1]): # to create the different camera views
    # set camera's x and y positions
    sqrt_mag = 2.1
    eye_x = sqrt_mag*np.sin(t)
    eye_y = sqrt_mag*np.cos(t)
    # generate base figure
    pad = -0.05
    subplot_specs = {'type': 'surface', 'l':pad, 'r':pad, 't':pad, 'b':pad}
    titles = [r'$\huge{p=1/2}$', r'$\huge{p=1}$', r'$\huge{p=2}$', r'$\huge{p=\infty}$']
    fig = make_subplots(rows=2, cols=2,
        specs=[[subplot_specs, subplot_specs], [subplot_specs, subplot_specs]],
        subplot_titles=titles,
        horizontal_spacing = 0.0, vertical_spacing = 0.0)
    # ------------------------------------------------------
    #  MAKE BALLS

    # L2 ball
    u, v = np.mgrid[0:2*np.pi:500j, 0:2*np.pi:500j]
    x2 = np.sin(u) * np.cos(v)
    y2 = np.sin(u) * np.sin(v)
    z2 = np.cos(u)
    fig.add_trace(go.Surface(x=x2, y=y2, z=z2, opacity=0.5, colorscale="Bluered", showscale=False), row=2, col=1)

    # L1 ball by normalization
    rho = 1/(np.abs(x2) + np.abs(y2) + np.abs(z2))
    x1 = rho * np.sin(u) * np.cos(v)
    y1 = rho * np.sin(u) * np.sin(v)
    z1 = rho * np.cos(u)
    fig.add_trace(go.Surface(x=x1, y=y1, z=z1, opacity=0.5, colorscale="Bluered", showscale=False), row=1, col=2)

    # Linf ball by normalization
    M = np.concatenate((x2[:,:,np.newaxis], y2[:,:,np.newaxis], z2[:,:,np.newaxis]), axis=2)
    rho = 1/np.max(np.abs(M),axis=2)
    xinf = rho * np.sin(u) * np.cos(v)
    yinf = rho * np.sin(u) * np.sin(v)
    zinf = rho * np.cos(u)
    fig.add_trace(go.Surface(x=xinf, y=yinf, z=zinf, opacity=0.5, colorscale="Bluered", showscale=False), row=2, col=2)

    # Lhalf ball by normalization
    rho = 1/(np.sqrt(np.abs(x2))+np.sqrt(np.abs(y2))+np.sqrt(np.abs(z2)))**2
    xhalf = rho * np.sin(u) * np.cos(v)
    yhalf = rho * np.sin(u) * np.sin(v)
    zhalf = rho * np.cos(u)
    fig.add_trace(go.Surface(x=xhalf, y=yhalf, z=zhalf, opacity=0.5, colorscale="Bluered", showscale=False), row=1, col=1)

    # ------------------------------------------------------
    # format subplots
    axis_dict = dict(tickvals=[-1,0,1], showticklabels=False, showgrid=True, gridcolor="gray", showbackground=False)
    scene_dict = dict(xaxis=axis_dict, yaxis=axis_dict, zaxis=axis_dict,
        xaxis_title='', yaxis_title='', zaxis_title='', # no axis labels
        camera=dict(eye=dict(x=eye_x, y=eye_y, z=1.2))) # set camera position (zoom+angle)
    # format complete image
    the_margin = go.layout.Margin(l=0, r=0, b=0, t=80) # get rid of white margins
    the_title = r'$\huge{\text{Unit} \hspace{0.5em} p\text{-balls} \hspace{0.5em} \text{in} \hspace{0.5em} \mathbb{R}^3}$'
    title_x = 0.52
    fig.update_layout(scene1=scene_dict, scene2=scene_dict, scene3=scene_dict, scene4=scene_dict,
        width=1000, height=1000, margin=the_margin, title_text=the_title, title_font_size=30, title_x=title_x, title_y=0.95)
    # make the titles closer to the plots
    fig['layout']['annotations'][0]['y'] = 0.93
    fig['layout']['annotations'][1]['y'] = 0.93
    fig['layout']['annotations'][2]['y'] = 0.47
    fig['layout']['annotations'][3]['y'] = 0.47
    # save this frame
    save_name = 'norm_balls_figures/balls_fig_' + str(idx_t+1) + '.png'
    fig.write_image(save_name)
    print('Saved figure', idx_t+1, '/', len(t_vec)-1)
