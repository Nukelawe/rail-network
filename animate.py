# This is the main file to generate the rail network animations
#
# Params:
# zoom_speed - change this number to alter the zoom in and out speed
#

import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from matplotlib.colors import to_hex
from matplotlib.patches import Circle
from network import Network
from scipy.ndimage import rotate
from scipy.special import betainc, beta
import matplotlib
from PIL import Image
import itertools

# initialize network from json file
inputfile = "random_stations_graph.json"
with open(inputfile, "r") as file:
    data = json.loads(file.read())
    nodes = data["nodes"]
    edges = data["edges"]
network = Network(nodes, edges)

fontsize_out = 30 # label font size in zoomed out scenes
fontsize_in = 60 # label font size in zoomed in scenes
stationsize = 10 # size of stations
intersectionsize = 10 # size of intersections
linewidth = stationsize / 6 # width of lines
label_offset = np.array([6,6])
cartsize = .03 * stationsize
zi = .07 # zoomed in level
zo = 1 # zoomed out level
blink_color = np.array([.5,.5,1.])
black = np.array([0.,0.,0.])
r = 10 # image rescale factor
pin = 2.0 # pin is the ease in factor
pout = 2.0 # pout is the ease out factor
num_blinks = 3 # number of times stations or intersections blink
blink_speed = 0.8 # time to blink once in seconds

# data limits for the zoomed out view
def zoom_limits(s, focus=None):
    zoom_out = np.array([[0, 0], [192, 108]])
    if focus is None: focus = np.mean(zoom_out, axis=0)
    s = betainc(pin,pout,s) / betainc(pin,pout,1)
    z = zo * (1-s) + zi * s
    A = zi * np.array([[.5,-.5], [.5, .5]])
    zoom_in = zi * np.array([[.5,-.5], [.5, .5]]) @ zoom_out + focus
    return zoom_in * s + zoom_out * (1-s), z

fps = 60 # frames per second
movement_speed = 3 # number of seconds to move from left to right accross the screen
rotation_speed = .5 # number of seconds to rotate pi radians
zoom_speed = 3.0 # number of seconds to zoom

def save_anim(filename, anim):
    print("Saving animation to file " + filename)
    anim.save(filename, fps=fps, dpi=100)

# set the zoom level of the given axis object
def set_zoom(ax, level, artists, focus=None):
    limits, z = zoom_limits(level, focus=focus)
    ax.set_xlim(limits[:,0])
    ax.set_ylim(limits[:,1])
    for marker in artists["stations"]:
        marker.set_markersize(stationsize / z)
    for marker in artists["intersections"]:
        marker.set_markersize(intersectionsize / z)
    for marker in artists["stations"] + artists["intersections"]:
        marker.set_markeredgewidth(linewidth / z)
    for line in artists["edges"]:
        line.set_linewidth(linewidth / z)
    for label in artists["nodelabels"]:
        label.set_fontsize(fontsize_out)
        label.xyann = label_offset / z
    return z

def plot_network(districts=True, annotate=True):
    fig, ax = network.plot(fig=plt.figure(figsize=(19.2,10.8)))
    lines = network.plot_edges(fig)
    stations,intersections,labels = network.plot_nodes(fig,
            districts=districts, annotate=annotate)
    return fig, ax, {"stations":stations,
            "intersections":intersections,
            "nodelabels":labels,
            "edges":lines}

#cart_img = plt.imread("images/chest_minecart_projection.png")
cart_img = Image.open("images/chest_minecart_projection.png")
cart_npimg = np.array(cart_img)
width,height = cart_img.size
cart_img_large = cart_img.resize((width*r,height*r), resample=Image.Resampling.BOX)
cart_npimg_large = np.array(cart_img_large)
def plot_cart(ax, pos, angle, addr):
    # cart image
    img = OffsetImage(rotate(cart_npimg_large, angle-90), zoom=cartsize*(.05/zi))
    box = AnnotationBbox(img, pos, frameon=False)
    # destination address annotation
    label = AnnotationBbox(
            TextArea(addr, multilinebaseline=True,
                textprops={"color":"white", "fontname":"sans",
                    "fontweight":"regular", "fontsize":fontsize_in,
                    "horizontalalignment":"center", "verticalalignment":"baseline",
                    "fontstretch":1000, "usetex":True}),
            pos, frameon=False, zorder=101,
            pad=0, box_alignment=(.50,.58))
    # outline for the destination address
    outline = Circle(pos, radius=.3, edgecolor="None",
            facecolor="black", alpha=.4, linewidth=.1*linewidth, zorder=100)
    return ax.add_artist(box), ax.add_artist(label), ax.add_artist(outline)

def zoom_animation(filename, focus, backwards=False):
    fig,ax,artists = plot_network(districts=False, annotate=True)
    if isinstance(focus, str):
        node = nodes[focus]
        focus = np.array([node["x"], node["z"]])

    def update(frame):
        if not backwards: frame = 1 - frame
        set_zoom(ax, 1-frame, artists, focus=focus)
        return list(itertools.chain(*artists.values()))

    frames = np.linspace(0, 1, int(fps * zoom_speed))
    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    save_anim(filename, anim)

def plot_routing_table(ax):
    # destination address annotation
    table_text = "\\renewcommand{\\arraystretch}{.9} \
            \\begin{tabular}{|c|c|} \
            \\hline \
            address & direction\\\\ \
            \\hline \
            0 & west \\\\ \
            1 & south \\\\ \
            2 & west \\\\ \
            3 & north \\\\ \
            4 & south \\\\ \
            5 & east \\\\ \
            6 & east \\\\ \
            7 & south \\\\ \
            8 & east \\\\ \
            \\vdots & \\vdots \\\\ \
            \\hline \
            \\end{tabular}"
    textArea = TextArea(table_text, multilinebaseline=True,
            textprops={"color":"white", "fontname":"sans",
                "fontweight":"regular", "fontsize":fontsize_out,
                "horizontalalignment":"left", "verticalalignment":"center",
                "fontstretch":1000, "usetex":True, "backgroundcolor":"black"})
    textArea._text.set_bbox(dict(facecolor='black', alpha=0.5, edgecolor="none"))
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    pad = .1 * (ylim[1] - ylim[0])
    pos = [xlim[0]+pad, ylim[1]-pad]
    table = AnnotationBbox(
            textArea, pos, frameon=False, zorder=101,
            pad=.0, box_alignment=(0,1), alpha=.5)
    return ax.add_artist(table)

def blink_animation(filename, annotate=True):
    fig,ax,artists = plot_network(districts=False, annotate=annotate)
    set_zoom(ax, 0, artists)
    markers = artists["stations"]
    if annotate: markers = artists["intersections"]

    def update(frame):
        s = np.sin(frame)**2
        color = to_hex(s * blink_color + (1-s) * black)
        for marker in markers:
            marker.set_markersize(stationsize * (1 + s))
            marker.set_markerfacecolor(color)
            marker.set_zorder(100)
        return markers

    frames = np.linspace(0, num_blinks*np.pi, int(blink_speed * fps * num_blinks))
    anim = FuncAnimation(fig, update, blit=True, frames=frames)
    save_anim(filename, anim)

directions = {"east": 0, "west": 180, "north": 90, "south": -90}
def rotate_animation(filename, focus, angle0, angle1, addr, districts=False):
    if not districts: addr = network.flatlabels[addr]
    node = nodes[focus]
    pos = np.array([node["x"], node["z"]])
    if isinstance(angle0, str): angle0 = directions[angle0]
    if isinstance(angle1, str): angle1 = directions[angle1]

    fig,ax,nw_artists = plot_network(districts, annotate=True)
    set_zoom(ax, 1, nw_artists, focus=pos)
    cart,label,outline = plot_cart(ax, pos, angle1, addr)
    box = cart.get_children()[0]

    def update(frame):
        angle = angle1 * frame + angle0 * (1-frame)
        cart_img_large.rotate(angle-90)
        cart_npimg_large = np.array(cart_img_large)
        box.set_data(rotate(cart_img_large, angle-90))
        return cart,

    numframes = fps * rotation_speed * np.amax(angle1 - angle0) / 180
    numframes = np.ceil(numframes).astype(int)
    frames = np.linspace(0, 1, numframes)
    anim = FuncAnimation(fig, update, blit=True, frames=frames)
    save_anim(filename, anim)

def add_cart_label(filename, p0, dirn, addr, districts=False, center=None):
    intersection = nodes[addr]["intersection"]
    if not districts: addr = network.flatlabels[addr]
    if isinstance(p0, str):
        node = nodes[p0]
        p0 = np.array([node["x"], node["z"]])
    if center is None: center = p0

    # plot the network
    fig,ax,nw_artists = plot_network(districts, annotate=True)
    # zoom onto center
    set_zoom(ax, 1, nw_artists, focus=center)

    angle = directions[dirn]
    pmid = np.array([np.mean(ax.get_xlim()), np.mean(ax.get_ylim())])

    cart,label,outline = plot_cart(ax, pmid, angle, addr)
    def update(frame):
        if frame == 1:
            label.set_visible(True)
            outline.set_visible(True)
        else:
            label.set_visible(False)
            outline.set_visible(False)
        return label,outline

    init = lambda:list(itertools.chain(*nw_artists.values()))

    anim = FuncAnimation(fig, update, init_func=init, blit=True, frames=[0,1])
    save_anim(filename, anim)

def move_cart(filename, p0, dirn, addr, backwards=False, districts=False, center=None,
        routing_table=False):
    intersection = nodes[addr]["intersection"]
    if not districts: addr = network.flatlabels[addr]
    if isinstance(p0, str):
        node = nodes[p0]
        p0 = np.array([node["x"], node["z"]])
    if center is None: center = p0

    # plot the network
    fig,ax,nw_artists = plot_network(districts, annotate=True)
    # zoom onto center
    set_zoom(ax, 1, nw_artists, focus=center)

    angle = directions[dirn]
    # get size of the currently visible section of the map
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    size = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0]])
    # movement direction
    dirn = np.array([np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)])
    p1 = p0 + .7 * size * dirn
    if backwards:
        p0,p1 = p1,p0
        angle *= -1
    pmid = np.array([np.mean(xlim), np.mean(ylim)])

    table = plot_routing_table(ax)
    cart,label,outline = plot_cart(ax, pmid, angle, addr)
    def update(frame):
        if frame < .05 or not routing_table:
            table.set_visible(False)
        else:
            table.set_visible(True)
        cart.xyann = p1 * frame + p0 * (1-frame)
        label.xyann = cart.xyann
        outline.center = cart.xyann
        return cart,label,outline,table

    init = lambda:list(itertools.chain(*nw_artists.values()))

    numframes = fps * movement_speed * np.amax(np.abs(p0-p1)) / size[0]
    numframes = np.ceil(numframes).astype(int)
    frames = np.linspace(0, 1, numframes)
    anim = FuncAnimation(fig, update, init_func=init, blit=True, frames=frames)
    save_anim(filename, anim)

def routing_animation(filename, focus, dirn, addr, districts=False):
    fig,ax,nw_artists = plot_network(districts, annotate=True)
    if isinstance(focus, str):
        node = nodes[focus]
        focus = np.array([node["x"], node["z"]])
    set_zoom(ax, 1, nw_artists, focus=focus)
    table = plot_routing_table(ax)
    def update(frame):
        return [table]
    anim = FuncAnimation(fig, update, blit=True, frames=[0])
    save_anim(filename, anim)

if __name__ =="__main__":
    blink_animation("blink_stations.mp4", annotate=False)
    blink_animation("blink_intersections.mp4", annotate=True)

    addr = "1.4"
    add_cart_label("station1.mp4", "1.3", "south", addr)
    move_cart("station2.mp4", "1.3", "south", addr)
    move_cart("intersection1.mp4", "1.0i", "north", addr, backwards=True, routing_table=True)
    move_cart("intersection3.mp4", "1.0i", "east", addr)

    zoom_animation("zoom1.mp4", "1.3")
    zoom_animation("zoom2.mp4", "1.3", backwards=True)
    zoom_animation("zoom3.mp4", "1.0i")
    zoom_animation("zoom4.mp4", "1.0i", backwards=True)

    rotate_animation("rotate1.mp4", "1.0i", "south", "east", addr)