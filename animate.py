# This is the main file to generate the rail network animations
#
# Params:
# zoom_speed - change this number to alter the zoom in and out speed
#

import numpy as np
import json, sys, itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from matplotlib.colors import to_hex
from matplotlib.patches import Circle
from matplotlib.table import Table
from network import Network
from scipy.ndimage import rotate
from scipy.special import betainc, beta
from PIL import Image

# initialize network from json file
inputfile = "random_stations_graph.json"
with open(inputfile, "r") as file:
    data = json.loads(file.read())
    nodes = data["nodes"]
    edges = data["edges"]
    routing_tables = data["routing_tables"]
network = Network(nodes, edges)

fontsize_out = 25 # label font size in zoomed out scenes
fontsize_in = 60 # label font size in zoomed in scenes
stationsize = 10 # size of stations
intersectionsize = 10 # size of intersections
linewidth = stationsize / 6 # width of lines
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
max_table_size = 8 # number of routing table rows to display

fps = 60 # frames per second
movement_speed = 3 # number of seconds to move from left to right accross the screen
rotation_speed = .5 # number of seconds to rotate pi radians
zoom_speed = 3.0 # number of seconds to zoom

cart_img = Image.open("images/chest_minecart_projection.png")
cart_npimg = np.array(cart_img)
width,height = cart_img.size
cart_img_large = cart_img.resize((width*r,height*r), resample=Image.Resampling.BOX)
cart_npimg_large = np.array(cart_img_large)

directions = {
    "east": 0,
    "west": 180,
    "north": 90,
    "south": -90,
    "south-west": -135,
}

class Animation():
    def __init__(self, districts=False, annotate=True):
        self.artists = {}
        self.fig = plt.figure(figsize=(19.2,10.8))
        self.ax = self.fig.gca()
        self.districts = districts
        self.annotate = annotate
        self.plot_network()
        self.set_zoom(0, None)

    def plot_network(self):
        network.plot(fig=self.fig)
        self.artists["lines"] = network.plot_edges(self.fig)
        s,i,l = network.plot_nodes(fig=self.fig, districts=self.districts,
                annotate=self.annotate)
        self.artists["stations"] = s
        self.artists["intersections"] = i
        self.artists["nodelabels"] = l

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
                        "horizontalalignment":"center", "verticalalignment":"baseline"}),
                pos, frameon=False, zorder=101, fontsize=fontsize_in,
                pad=0, box_alignment=(.50,.58))

        # outline for the destination address
        self.artists["dest_outline"] = Circle(pos, radius=.3, edgecolor="None",
                facecolor="black", alpha=.4, linewidth=.1*linewidth, zorder=100)

        for artist in ["cart", "destlabel", "dest_outline"]:
            self.ax.add_artist(self.artists[artist])

    def set_zoom(self, level, focus):
        zoom_out = np.array([[0, 0], [192, 108]])
        if isinstance(focus, str):
            node = nodes[focus]
            focus = np.array([node["x"], node["z"]])
        elif focus is None:
            focus = np.mean(zoom_out, axis=0)
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
        ms = self.artists["stations"][0].get_markersize()
        for label in self.artists["nodelabels"]:
            label.get_children()[0]._text.set_fontsize(fontsize)
            radius = self.artists["stations"][0].get_markersize() / 2 + .5 * linewidth * z
            offset = (radius + linewidth * z + .5 * fontsize) / np.sqrt(2)
            label.xyann = [1.3*offset, offset]
        return z

    def plot_routing_table(self, addr):
        # construct the routing table
        table = routing_tables[addr]
        labels = np.array(list(table.keys()))
        faddr = np.array([network.flatlabels[label] for label in labels])
        #gaddr = np.array([Network.split_address(label)[0] for label in labels])
        #laddr = np.array([Network.split_address(label)[1] for label in labels])
        sorted_indices = np.argsort(faddr)
        labels = labels[sorted_indices]
        faddr = faddr[sorted_indices]
        routingdirs = np.array([table[label] for label in labels])
        faddr = np.reshape(faddr, [faddr.size, 1])
        labels = np.reshape(labels, [labels.size, 1])
        routingdirs = np.reshape(routingdirs, [labels.size, 1])
        cellText = np.concatenate((faddr, routingdirs), axis=1)
        cellText = cellText[:max_table_size,:] # limit the table display size
        cellText = np.concatenate((cellText, [["$\\vdots$ ","$\\vdots$ "]]), axis=0)

        xlim = self.ax.get_xlim(); ylim = self.ax.get_ylim()
        pad = .1 * (ylim[1] - ylim[0])
        xmid = np.mean(xlim)
        ymid = np.mean(ylim)

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

def zoom_animation(focus, backwards=False):
    anim = Animation()
    artists = [anim.artists["stations"], anim.artists["intersections"],
            anim.artists["lines"], anim.artists["nodelabels"]]
    artists = list(itertools.chain(*artists))
    def update(frame):
        if not backwards: frame = 1 - frame
        anim.set_zoom(1-frame, focus)
        return artists

    frames = np.linspace(0, 1, int(fps * zoom_speed))
    return FuncAnimation(anim.fig, update, frames=frames, blit=True)

def blink_markers(markertype, **kwargs):
    anim = Animation(**kwargs)
    markers = anim.artists[markertype]
    def update(frame):
        s = np.sin(frame)**2
        color = to_hex(s * blink_color + (1-s) * black)
        for marker in markers:
            marker.set_markersize(stationsize * (1 + s))
            marker.set_markerfacecolor(color)
            marker.set_zorder(100)
        return markers
    frames = np.linspace(0, num_blinks*np.pi, int(blink_speed * fps * num_blinks))
    return FuncAnimation(anim.fig, update, blit=True, frames=frames)

def blink_labels(labeltype, **kwargs):
    anim = Animation(districts=True, **kwargs)
    def update(frame):
        s = np.sin(frame)**2
        color = to_hex(s * blink_color + (1-s) * black)
        for label,markerdict in network.markers.items():
            if labeltype not in markerdict: continue
            address = markerdict[labeltype].get_children()[0].get_children()[0]
            address.set_fontsize(fontsize_out * (1 + s))
            markerdict[labeltype].set_zorder(100)
        return anim.artists["nodelabels"]
    frames = np.linspace(0, num_blinks*np.pi, int(blink_speed * fps * num_blinks))
    return FuncAnimation(anim.fig, update, blit=True, frames=frames)

def rotate_animation(focus, dirn0, dirn1, addr):
    anim = Animation()
    anim.set_zoom(1, focus)
    anim.plot_cart(dirn0, addr)
    anim.plot_routing_table(focus)

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

    numframes = int(fps * rotation_speed * np.abs(angle1 - angle0) / 180)
    frames = np.linspace(0, 1, numframes)
    frames = np.concatenate(([0], frames))
    return FuncAnimation(anim.fig, update, blit=True, frames=frames)

def add_cart_label(focus, dirn, addr):
    anim = Animation()
    anim.set_zoom(1, focus)
    anim.plot_cart(dirn, addr)
    anim.artists["destlabel"].set_visible(False)
    anim.artists["dest_outline"].set_visible(False)
    return FuncAnimation(anim.fig, lambda frame: [], blit=True, frames=[0])

def move_cart(
        focus, # where to focus the camera
        p1, # direction to move the cart
        addr, # destination address of the cart
        backwards=False, # should the movement be out->in instead of in->out
        routing_table=False): # should the routing table be drawn
    # by default the cart starts from the middle of the screen and moves to the specified
    # end point. If the end point is a direction, then the cart moves off the screen in
    # that direction.
    anim = Animation()
    anim.set_zoom(1, focus)
    if routing_table: anim.plot_routing_table(focus)

    # get size of the currently visible section of the map
    xlim = anim.ax.get_xlim(); ylim = anim.ax.get_ylim()
    size = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0]])
    node = nodes[focus]
    p0 = np.array([node["x"], node["z"]]) # starting position
    if p1 in directions:
        angle = directions[p1] # cart's angle
        # unit vector in the direction of movement
        d = np.array([np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)])
        p1 = p0 + .7 * size * d # position to move to (outside view)
    elif p1 in nodes:
        node = nodes[p1]
        marker = network.markers[p1]["station"]
        p1 = marker.get_xydata()[0]
        angle = directions["south-west"]
    if backwards:
        p0,p1 = p1,p0
        angle += 180
    anim.plot_cart(angle, addr)

    cart = anim.artists["cart"]
    label = anim.artists["destlabel"]
    outline = anim.artists["dest_outline"]
    def update(frame):
        cart.xyann = p1 * frame + p0 * (1-frame)
        label.xyann = cart.xyann
        outline.center = cart.xyann
        return cart, label, outline

    dist = np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
    numframes = int(fps * movement_speed * dist / size[0])
    frames = np.concatenate(([0],np.linspace(0, 1, numframes)))
    return FuncAnimation(anim.fig, update, blit=True, frames=frames)

def routing_setup(focus, dirn, addr):
    anim = Animation()
    anim.set_zoom(1, focus)
    anim.plot_cart(dirn, addr)
    anim.plot_routing_table(focus)
    return anim

def routing_animate(anim, update):
    frames = np.linspace(0, num_blinks*np.pi, int(blink_speed * fps * num_blinks))
    # this is a dirty fix where the first frame shows big text in the table
    frames = np.concatenate(([0],frames))
    return FuncAnimation(anim.fig, update, blit=True, frames=frames)

def blink_table(focus, dirn, addr):
    anim = routing_setup(focus, dirn, addr)
    def update(frame):
        s = np.sin(frame)**2
        cells = anim.artists["edges_table"].get_celld()
        row = -1
        for key in cells:
            if key[1] == 1: continue
            address = cells[key].get_text()._text
            if address == str(network.flatlabels[addr]):
                row = key[0]
        cells = anim.artists["color_table"].get_celld()
        for cell in [cells[row,0], cells[row,1]]:
            cell.set_facecolor(to_hex(s * blink_color + (1-s) * black))
        return anim.artists["color_table"],
    return routing_animate(anim, update)

def blink_cart(focus, dirn, addr, districts=False):
    anim = routing_setup(focus, dirn, addr)
    def update(frame):
        s = np.sin(frame)**2
        outline = anim.artists["dest_outline"]
        outline.set_facecolor(to_hex(s * blink_color + (1-s) * black))
        return outline,
    return routing_animate(anim, update)

animation_instructions = {
    "zoom1":
        lambda: zoom_animation("1.3"),
    "zoom2":
        lambda: zoom_animation("1.3", backwards=True),
    "zoom3":
        lambda: zoom_animation("1.0i"),
    "zoom4":
        lambda: zoom_animation("1.0i", backwards=True),
    "zoom5":
        lambda: zoom_animation("1.4"),
    "zoom6":
        lambda: zoom_animation("1.4", backwards=True),
    "station1":
        lambda: add_cart_label("1.3", "south", "1.4"),
    "station2":
        lambda: move_cart("1.3", "south", "1.4"),
    "intersection1":
        lambda: move_cart("1.0i", "north", "1.4", backwards=True),
    "intersection1_routingtable":
        lambda: move_cart("1.0i", "north", "1.4", backwards=True,
            routing_table=True),
    "intersection2":
        lambda: move_cart("1.0i", "east", "1.4"),
    "intersection2_routingtable":
        lambda: move_cart("1.0i", "east", "1.4", routing_table=True),
    "intersection3":
        lambda: move_cart("1.4", "west", "1.4", backwards=True),
    "intersection4":
        lambda: move_cart("1.4", "1.4", "1.4", routing_table=True),
    "blink_stations":
        lambda: blink_markers("stations", annotate=False),
    "blink_intersections":
        lambda: blink_markers("intersections"),
    "rotate1":
        lambda: rotate_animation("1.0i", "south", "east", "1.4"),
    "rotate2":
        lambda: rotate_animation("1.4", "east", "south-west", "1.4"),
    "routing1":
        lambda: blink_cart("1.0i", "south", "1.4"),
    "routing2":
        lambda: blink_table("1.0i", "south", "1.4"),
    "routing3":
        lambda: blink_cart("1.4", "east", "1.4"),
    "routing4":
        lambda: blink_table("1.4", "east", "1.4"),
    "blink_local":
        lambda: blink_labels("laddr"),
    "blink_global":
        lambda: blink_labels("gaddr"),
}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Incorrect number of arguments,\
                run as 'python network.py <filename>'")
    filename = sys.argv[1] # name of the file to save the animation to
    name = filename.split(".")[0] # filename without the type extension
    if name not in animation_instructions:
        raise Exception("The instructions for the creating the file {}\
                are missing.".format(filename))
    print("Saving animation to file " + filename)
    anim = animation_instructions[name]()
    anim.save(filename, fps=fps, dpi=100)
