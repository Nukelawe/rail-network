import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from network import Network


inputfile = "random_stations_graph.json"
with open(inputfile, "r") as file:
    data = json.loads(file.read())
    nodes = data["nodes"]
    edges = data["edges"]
network = Network(nodes, edges)

fig = plt.figure(figsize=(19.20,10.80), dpi=1)
ax = fig.gca()

start_limits = np.array([[0, 0], [192, 108]])
center = np.array([72, 72])

def zoomed_limits(start_limits, focus, s):
    z0 = .05 # maximum zoom level
    z = 1 + (z0 - 1) * s
    A = z * np.array([
        [1-.5*s,  -.5*s],
        [  .5*s, 1-.5*s]
    ])
    limits = A @ start_limits + (1-(1-s)*z) * focus
    return limits

def init():
    ax.set_xlim(start_limits[:,0])
    ax.set_ylim(start_limits[:,1])
    ax.set_aspect('equal', adjustable='box')
    return network.plot(fig)

def update(frame):
    limits = zoomed_limits(start_limits, center, frame)
    ax.set_xlim(limits[:,0])
    ax.set_ylim(limits[:,1])
    return []

anim = FuncAnimation(fig, update, frames=np.linspace(0, 1, 50),
        init_func=init, blit=True)
filename = "output.mp4"
print("Saving animation to file " + filename)
anim.save(filename, fps=30, dpi=100)
