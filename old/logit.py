from typing import Iterable
from matplotlib.artist import Artist
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

def f(x):
    return 1 / ( 1 + np.exp(2 + -0.07*x) )

x = np.arange(1, 100, 1)
y = f(x)

fig, ax = plt.subplots()
line2d = ax.plot(x[0], y[0])[0]
ax.set(xlim=[0, 100], ylim=[0, 1], xlabel='x', ylabel='y')

def update(frame) -> Iterable[Artist]:
    dx = x[:frame]
    dy = y[:frame]
    line2d.set_data(dx, dy)
    return [line2d]

ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=40, repeat_delay=2000)

plt.show()
