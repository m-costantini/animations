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
import numpy as np

# generate parametric rotated ellipse for plotting

theta = pi/4 # tilting angle = 45 degrees
t = np.linspace(0,2*pi,60) # parameter of the curve

Rx = 1
Ry = 0.5

x = Rx*np.cos(t)*np.cos(theta) - Ry*np.sin(t)*np.sin(theta)
y = Rx*np.cos(t)*np.sin(theta) + Ry*np.sin(t)*np.cos(theta)

fig = px.line(x=x, y=y, title='A curve')
fig.show()
