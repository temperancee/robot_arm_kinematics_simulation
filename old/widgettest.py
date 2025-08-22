import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Data setup
x = np.arange(1, 100, 1)
z = 8
y = 2*x+3 - z

# Axes setup
fig = plt.figure()
ax = fig.add_subplot()
ax.set(xlim=[-10, 100], ylim=[-50, 200], xlabel='x', ylabel='y')

# Initial plot
line = ax.plot(x, y)[0]

def slider_update(val):
    global z
    z=val
    update_plot()

def update_plot():
    y = 2*x+3 - z
    line.set_data(x, y)
    fig.canvas.draw_idle()
    print(x,y,z)

# Widgets
ax_slider = fig.add_axes((0.25, 0.01 + 0.03, 0.65, 0.02))
s = Slider(ax_slider, f"Z", -20, 20, valinit=8)
s.on_changed(slider_update)


plt.show()
