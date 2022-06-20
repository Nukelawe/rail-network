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
from matplotlib.table import Table
from network import Network
from scipy.ndimage import rotate
from scipy.special import betainc, beta
import matplotlib
from PIL import Image
import itertools

plt.rc("text.latex", preamble=r"\usepackage{colortbl}\usepackage{xcolor}")

# initialize network from json file
inputfile = "random_stations_graph.json"
with open(inputfile, "r") as file:
    data = json.loads(file.read())
    nodes = data["nodes"]
    edges = data["edges"]
network = Network(nodes, edges)

fontsize_out = 25 # label font size in zoomed out scenes
fontsize_in = 60 # label font size in zoomed in scenes
stationsize = 10 # size of stations
intersectionsize = 10 # size of intersections
linewidth = stationsize / 6 # width of lines
label_offset = np.array([6,6])
cartsize = stationsize
zi = 14 # zoomed in level
zo = 1 # zoomed out level
blink_color = np.array([.4,1.,.4])
black = np.array([0.,0.,0.])
r = 10 # image rescale factor
pin = 2.0 # pin is the ease in factor
pout = 2.0 # pout is the ease out factor
num_blinks = 3 # number of times stations or intersections blink
blink_speed = 0.8 # time to blink once in seconds

fps = 60 # frames per second
movement_speed = 3 # number of seconds to move from left to right accross the screen
rotation_speed = .5 # number of seconds to rotate pi radians
zoom_speed = 3.0 # number of seconds to zoom

cart_img = Image.open("images/chest_minecart_projection.png")
cart_npimg = np.array(cart_img)
width,height = cart_img.size
cart_img_large = cart_img.resize((width*r,height*r), resample=Image.Resampling.BOX)
cart_npimg_large = np.array(cart_img_large)

directions = {"east": 0, "west": 180, "north": 90, "south": -90}

class Animation():
    def __init__(self):
        self.artists = {}
        self.anim = None
        self.fig = plt.figure(figsize=(19.2,10.8))
        self.ax = self.fig.gca()
        self.districts=False
        self.annotate=True

    def set_districts(b): self.districts=b
    def set_annotate(b): self.annotate=b

    def plot_network(self):
        network.plot(fig=self.fig)
        self.artists["lines"] = network.plot_edges(self.fig)
        s,i,l = network.plot_nodes(fig=self.fig, districts=self.districts,
                annotate=self.annotate)
        self.artists["stations"] = s
        self.artists["intersections"] = i
        self.artists["nodelabels"] = l
        for label in l:
            label.set_fontsize(fontsize_out)

    def plot_cart(self,
            dirn, # direction the cart is facing
            addr # destination address to draw on the cart
        ):
        angle = dirn
        if isinstance(dirn, str):
            angle = directions[dirn]
        xlim = self.ax.get_xlim(); ylim = self.ax.get_ylim()
        pos = np.array([np.mean(xlim), np.mean(ylim)])

        # cart image
        img = OffsetImage(rotate(cart_npimg_large, angle-90), zoom=.0015 * cartsize * zi)
        self.artists["cart"] = AnnotationBbox(img, pos, frameon=False)

        # destination address annotation
        if not self.districts: addr = network.flatlabels[addr]
        self.artists["destlabel"] = AnnotationBbox(
                TextArea(addr, multilinebaseline=True,
                    textprops={"color":"white", "fontname":"sans",
                        "fontweight":"regular", "fontsize":fontsize_in,
                        "horizontalalignment":"center", "verticalalignment":"baseline",
                        "fontstretch":1000, "usetex":True}),
                pos, frameon=False, zorder=101,
                pad=0, box_alignment=(.50,.58))

        # outline for the destination address
        self.artists["dest_outline"] = Circle(pos, radius=.3, edgecolor="None",
                facecolor="black", alpha=.4, linewidth=.1*linewidth, zorder=100)

        for artist in ["cart", "destlabel", "dest_outline"]:
            self.ax.add_artist(self.artists[artist])

    def set_zoom(self, level, focus):
        if isinstance(focus, str):
            node = nodes[focus]
            focus = np.array([node["x"], node["z"]])
        zoom_out = np.array([[0, 0], [192, 108]])
        s = betainc(pin,pout,level) / betainc(pin,pout,1)
        z = 1 / ((1-s) / zo + s / zi)
        zoom_in = np.array([[.5,-.5], [.5, .5]]) / zi @ zoom_out + focus
        limits = zoom_in * s + zoom_out * (1-s)
        self.ax.set_xlim(limits[:,0])
        self.ax.set_ylim(limits[:,1])
        for marker in self.artists["stations"]:
            marker.set_markersize(stationsize * z)
        for marker in self.artists["intersections"]:
            marker.set_markersize(intersectionsize * z)
        for marker in self.artists["stations"] + self.artists["intersections"]:
            marker.set_markeredgewidth(linewidth * z)
        for line in self.artists["lines"]:
            line.set_linewidth(linewidth * z)

        # bezier curve font size interpolation
        z0 = zo; z1 = fontsize_in / fontsize_out; z2 = zi
        f0 = fontsize_out; f1 = fontsize_in; f2 = fontsize_in;
        a = z0 - 2 * z1 + z2
        b = -2 * (z0 - z1)
        c = z0 - z
        t = .5 / a * (-b + np.sqrt(b**2 - 4 * a * c))
        fontsize = (1-t) * ((1-t) * f0 + t * f1) + t * ((1-t) * f1 + t * f2)
        for label in self.artists["nodelabels"]:
            label.set_fontsize(fontsize)
            label.xyann = label_offset * z
        return z

    def plot_routing_table(self):
        xlim = self.ax.get_xlim(); ylim = self.ax.get_ylim()
        pad = .1 * (ylim[1] - ylim[0])
        xmid = np.mean(xlim)
        ymid = np.mean(ylim)
        cellText = np.array([
                ["0","west" ],
                ["1","south"],
                ["2","west" ],
                ["3","north"],
                ["4","south"],
                ["5","east" ],
                ["6","east" ],
                ["7","south"],
                ["8","east" ],
                ["$\\vdots$ ","$\\vdots$ "]
        ])
        bbox = [.25,.57,.18,.95-.57] # bounding box of the table
        color_table = self.ax.table(cellText=cellText,
                colLabels=["", ""], bbox=bbox)
        edges_table = self.ax.table(cellText=cellText, cellLoc="center",
                colLabels=["address", "direction"], bbox=bbox)
        for col in np.arange(cellText.shape[1]):
            for row in np.arange(cellText.shape[0]+1):
                edges_table[row,col].get_text().set_fontsize(fontsize_out)
                edges_table[row,col].get_text().set_color("white")
                color_table[row,col].get_text()._text = ""
                edges_table[row,col].set_edgecolor("white")
                edges_table[row,col].set_facecolor("none")
                color_table[row,col].set_facecolor("black")
                if row == 0:
                    edges_table[row,col].visible_edges = "BRTL"
                elif row == cellText.shape[0]:
                    edges_table[row,col].visible_edges = "BRL"
                else:
                    edges_table[row,col].visible_edges = "RL"
        self.artists["edges_table"] = edges_table
        self.artists["color_table"] = color_table

    def save(self,
            filename, # name of the file to save the animation
            anim # animation object
        ):
        print("Saving animation to file " + filename)
        anim.save(filename, fps=fps, dpi=100)

def zoom_animation(filename, focus, backwards=False):
    anim = Animation()
    anim.plot_network()
    artists = [anim.artists["stations"], anim.artists["intersections"],
            anim.artists["lines"], anim.artists["nodelabels"]]
    artists = list(itertools.chain(*artists))
    def update(frame):
        if not backwards: frame = 1 - frame
        anim.set_zoom(1-frame, focus)
        return artists

    frames = np.linspace(0, 1, int(fps * zoom_speed))
    a = FuncAnimation(anim.fig, update, frames=frames, blit=True)
    anim.save(filename, a)

def blink_animation(filename, annotate=True):
    anim = Animation()
    anim.plot_network()

    markers = anim.artists["stations"]
    if annotate: markers = anim.artists["intersections"]
    def update(frame):
        s = np.sin(frame)**2
        color = to_hex(s * blink_color + (1-s) * black)
        for marker in markers:
            marker.set_markersize(stationsize * (1 + s))
            marker.set_markerfacecolor(color)
            marker.set_zorder(100)
        return markers
    frames = np.linspace(0, num_blinks*np.pi, int(blink_speed * fps * num_blinks))
    a = FuncAnimation(anim.fig, update, blit=True, frames=frames)
    anim.save(filename, a)

def rotate_animation(filename, focus, dirn0, dirn1, addr):
    anim = Animation()
    anim.plot_network()
    anim.set_zoom(1, focus)
    anim.plot_cart(dirn0, addr)
    anim.plot_routing_table()

    cart = anim.artists["cart"]
    label = anim.artists["destlabel"]
    outline = anim.artists["dest_outline"]
    box = cart.get_children()[0]
    angle0 = directions[dirn0]
    angle1 = directions[dirn1]
    def update(frame):
        angle = directions[dirn1] * frame + directions[dirn0] * (1-frame)
        cart_img_large.rotate(angle-90)
        cart_npimg_large = np.array(cart_img_large)
        box.set_data(rotate(cart_img_large, angle-90))
        return cart,

    numframes = int(fps * rotation_speed * np.amax(angle1 - angle0) / 180)
    frames = np.linspace(0, 1, numframes)
    a = FuncAnimation(anim.fig, update, blit=True, frames=frames)
    anim.save(filename, a)

def add_cart_label(filename, focus, dirn, addr):
    anim = Animation()
    anim.plot_network()
    anim.set_zoom(1, focus)
    anim.plot_cart(dirn, addr)
    anim.artists["destlabel"].set_visible(False)
    anim.artists["dest_outline"].set_visible(False)

    def update(frame): return []
    a = FuncAnimation(anim.fig, update, blit=True, frames=[0])
    anim.save(filename, a)

def move_cart(filename, focus, dirn, addr, backwards=False, center=None,
        routing_table=False):
    anim = Animation()
    anim.plot_network()
    anim.set_zoom(1, focus)
    anim.plot_routing_table()

    node = nodes[focus]
    p0 = np.array([node["x"], node["z"]])
    angle = directions[dirn]
    # get size of the currently visible section of the map
    xlim = anim.ax.get_xlim(); ylim = anim.ax.get_ylim()
    size = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0]])
    # movement direction
    d = np.array([np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)])
    p1 = p0 + .7 * size * d
    if backwards:
        p0,p1 = p1,p0
        angle *= -1
    pmid = np.array([np.mean(xlim), np.mean(ylim)])
    anim.plot_cart(dirn, addr)

    cart = anim.artists["cart"]
    label = anim.artists["destlabel"]
    outline = anim.artists["dest_outline"]
    def update(frame):
        if frame < .05 or not routing_table:
            anim.artists["color_table"].set_visible(False)
            anim.artists["edges_table"].set_visible(False)
        else:
            anim.artists["color_table"].set_visible(True)
            anim.artists["edges_table"].set_visible(True)
        cart.xyann = p1 * frame + p0 * (1-frame)
        label.xyann = cart.xyann
        outline.center = cart.xyann
        return [cart,label,outline,
                anim.artists["color_table"],anim.artists["edges_table"]]

    init = lambda:list(itertools.chain(*nw_artists.values()))

    numframes = int(fps * movement_speed * np.amax(np.abs(p0-p1)) / size[0])
    frames = np.linspace(0, 1, numframes)
    a = FuncAnimation(anim.fig, update, blit=True, frames=frames)
    anim.save(filename, a)

def plot_routing_intersection(dirn, focus, addr, districts=False):
    nw_artists = plot_network(districts, annotate=True)
    if not districts: addr = network.flatlabels[addr]
    if isinstance(focus, str):
        node = nodes[focus]
        focus = np.array([node["x"], node["z"]])

def blink_table(filename, focus, dirn, addr):
    anim = Animation()
    anim.plot_network()
    anim.set_zoom(1, focus)
    anim.plot_cart(dirn, addr)
    anim.plot_routing_table()

    table = anim.artists["color_table"]
    def update(frame):
        s = np.sin(frame)**2
        for cell in [table[6,0], table[6,1]]:
            cell.set_facecolor(to_hex(s * blink_color + (1-s) * black))
        return table,
    frames = np.linspace(0, num_blinks*np.pi, int(blink_speed * fps * num_blinks))
    a = FuncAnimation(anim.fig, update, blit=True, frames=frames)
    anim.save(filename, a)

def blink_cart(filename, focus, dirn, addr, districts=False):
    anim = Animation()
    anim.plot_network()
    anim.set_zoom(1, focus)
    anim.plot_cart(dirn, addr)
    anim.plot_routing_table()
    def update(frame):
        outline = anim.artists["dest_outline"]
        s = np.sin(frame)**2
        outline.set_facecolor(to_hex(s * blink_color + (1-s) * black))
        return outline,
    frames = np.linspace(0, num_blinks*np.pi, int(blink_speed * fps * num_blinks))
    a = FuncAnimation(anim.fig, update, blit=True, frames=frames)
    anim.save(filename, a)

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
    blink_cart("routing1.mp4", "1.0i", "south", addr)
    blink_table("routing2.mp4", "1.0i", "south", addr)
