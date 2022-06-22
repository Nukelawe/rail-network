import numpy as np
from voronoi_chebyshev import voronoi_chebyshev
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import json

banner_colors = {
        "white":"#ffffff", "light_gray":"#9c9d96", "gray":"#464f53", "black":"#1d1c21",
        "brown":"#835432", "red":"#b02e25", "orange":"#f8801d", "yellow":"#fed83a",
        "lime":"#80c61f", "green":"#5e7c17", "cyan":"#169d9d", "light_blue":"#3bb4da",
        "blue":"#3b44aa", "magenta":"#c74ebd", "purple":"#8931b8", "pink":"#f18ca9",
        }
district_colors = [
        "lime", "pink", "yellow", "cyan",
        "purple", "light_blue", "brown", "red",
        "orange", "blue", "green", "brown",
        "brown", "black", "gray", "light_gray"]
# style parameters
linecolor = "white" # color of lines
bgcolor = "black" # background color
debugcolor = "#555555"

line_width = 10/6 # width of lines
stationsize = 10 # size of stations
intersectionsize = 10 # size of intersections
fontsize_out = 20 # label font size in zoomed out scenes
fontsize_in = 60 # label font size in zoomed in scenes
label_offset = np.array([6,6])

# style arguments for lines
lineargs= {
        "linestyle":"solid",
        "marker":None,
        "color":linecolor,
        "linewidth":line_width
}; labelargs = {
        "textcoords":"offset points",
        "xytext":label_offset,
        "fontname":"sans serif",
        "fontweight":"regular",
        "fontsize":fontsize_out
}; nodeargs = {
        "markeredgewidth":line_width,
        "markersize":stationsize,
        "markerfacecolor":bgcolor,
        "markeredgecolor":linecolor,
        "linestyle":None
}

class Network:
    def split_address(label):
        addr = label.split(".")
        if len(addr) == 2 and addr[0].isnumeric():
            return addr[0],addr[1]
        return 0,0

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.markers = {} # label -> list of markers at that label
        self.districts = {} # district number -> list of node indices
        # numpy array of node coords
        self.points = np.empty(len(self.nodes), dtype=[("x",int), ("z",int)])
        labels = np.array(list(self.nodes.keys()))
        station_inds = [] # indices of station nodes
        for i,label in enumerate(labels):
            node = self.nodes[label]
            if node["station"]:
                station_inds.append(i)
            self.points[i] = (node["x"], node["z"])
            gaddr,_ = Network.split_address(label)
            if gaddr not in self.districts.keys():
                self.districts[gaddr] = []
            self.districts[gaddr].append(i)

        self.flatlabels = {} # label -> flattened station address
        station_points = self.points[station_inds]
        station_labels = labels[station_inds]
        station_points["z"] *= -1
        inds = np.argsort(station_points, order=("x", "z"))
        station_labels = station_labels[inds]
        for i,label in enumerate(station_labels):
            self.flatlabels[label] = i

        self.connections = {} # label -> number of incident edges to the node
        for node in self.nodes:
            self.connections[node] = 0
        for edge in self.edges:
            self.connections[edge["from"]] += 1
            self.connections[edge["to"]] += 1

        # check that the network is valid
        for label,node in self.nodes.items():
            if self.connections[label] > 1 and not node["intersection"]:
                raise Exception("Only intersections may have many incident edges. \
                        The non-intersection node '" + label + "' has " +
                        self.connections[label])
            if self.connections[label] > 4:
                raise Exception("A node may have at most 4 incident edges. \
                        Node '" + label + "' has " + self.connections[label])
            if not node["station"] and not node["intersection"]:
                raise Exception("A node must be either an intersection, station or both")

    # computes the voronoi diagram to visualize districts
    def district_boundaries(self):
        xmin = np.amin(self.points["x"])
        xmax = np.amax(self.points["x"])
        zmin = np.amin(self.points["z"])
        zmax = np.amax(self.points["z"])
        xwidth = xmax - xmin
        zwidth = zmax - zmin
        r = 2
        return voronoi_chebyshev(self.points, self.districts,
                xmin-r*xwidth, xmax+r*xwidth,
                zmin-r*zwidth, zmax+r*zwidth)

    ##
    ## Network plotting and styling
    ##
    def plot(self, fig=None, debug=False):
        if fig is None:
            fig = plt.figure(figsize=(19.20,10.87), dpi=129.05)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.debug = debug
        ax = fig.gca()
        ax.set_xlim((0, 192))
        ax.set_ylim((0, 108))
        if debug:
            granularity = 6
            ax.set_xticks(np.arange(0, 192, granularity))
            ax.set_yticks(np.arange(0, 108, granularity))
            ax.grid(color=debugcolor, linewidth=.5)
            ax.set_facecolor(bgcolor)
            ax.tick_params(axis="both", colors=debugcolor)
        else:
            ax.axis("off")
        fig.set_facecolor(bgcolor)
        ax.set_aspect("equal", adjustable="box")
        return fig, ax

    def plot_edges(self, fig=plt.gcf()):
        ax = fig.gca()
        edges = []
        for edge in self.edges:
            nodefrom = self.nodes[edge["from"]]
            nodeto = self.nodes[edge["to"]]
            edgetype = edge["type"]
            xmid = .5*(nodefrom["x"] + nodeto["x"])
            zmid = .5*(nodefrom["z"] + nodeto["z"])
            dx = np.abs(nodeto["x"] - nodefrom["x"])#; dx = 0
            dz = np.abs(nodeto["z"] - nodefrom["z"])#; dz = 0
            if edgetype == "zx":
                a = ax.plot([nodefrom["x"], nodefrom["x"], nodeto["x"]],
                        [nodefrom["z"], nodeto["z"],   nodeto["z"]],
                        **lineargs)
            elif edgetype == "xz":
                a = ax.plot([nodefrom["x"], nodeto["x"],   nodeto["x"]],
                        [nodefrom["z"], nodefrom["z"], nodeto["z"]],
                        **lineargs)
            elif edgetype == "xx":
                a = ax.plot([nodefrom["x"], xmid-.5*dz,    xmid+.5*dz,  nodeto["x"]],
                        [nodefrom["z"], nodefrom["z"], nodeto["z"], nodeto["z"]],
                        **lineargs)
            elif edgetype == "zz":
                a = ax.plot([nodefrom["x"], nodefrom["x"], nodeto["x"], nodeto["x"]],
                        [nodefrom["z"], zmid-.5*dx,    zmid+.5*dx,  nodeto["z"]],
                        **lineargs)
            if a is not None:
                edges.append(a[0])
        return edges

    def plot_nodes(self, fig=plt.gcf(), districts=True, annotate=True):
        stations = []
        intersections = []
        labels = []
        ax = fig.gca()
        trans = ax.transData
        trans_inv = ax.transData.inverted()
        offset = np.array([-7.5/72.*fig.dpi, -7.5/72.*fig.dpi])

        # marker size
        disp1 = trans.transform((.5,.5))
        disp0 = trans.transform((0,0))
        markersize = (disp1-disp0)[0]
        for label,node in self.nodes.items():
            self.markers[label] = {}
            if node["station"]:
                disp = trans.transform((node["x"],node["z"]))
                data = trans_inv.transform(disp)
                # draw a line connecting the intersection station to the intersection
                if node["intersection"]:
                    dataoffset = trans_inv.transform(disp + offset)
                    ax.plot([data[0], dataoffset[0]],
                            [data[1], dataoffset[1]], **lineargs)
                    data = dataoffset
                a = ax.plot(data[0], data[1], marker="o", **nodeargs)[0]
                self.markers[label]["station"] = a
                stations.append(a)

                annotation = label
                if not districts: annotation = self.flatlabels[label]
                if annotate:
                    a = ax.annotate(annotation, (node["x"], node["z"]), color=linecolor,
                            **labelargs)
                    labels.append(a)

            if node["intersection"]:
                a = ax.plot(node["x"], node["z"], marker="D", **nodeargs)[0]
                self.markers[label]["intersection"] = a
                intersections.append(a)
                if self.debug and annotate:
                    a = ax.annotate(label, (node["x"], node["z"]),
                            color=debugcolor, **labelargs)
                    labels.append(a)
        if districts: self._plot_districts(ax)
        return stations, intersections, labels

    def _plot_districts(self, ax):
        # color the map
        cells = self.district_boundaries()
        for district in cells:
            color = banner_colors[district_colors[int(district)]]
            patch = plt.Polygon(cells[district], linestyle=None, facecolor=color,
                    alpha=.25, edgecolor="None")
            ax.add_patch(patch)

if __name__ =="__main__":
    inputfile = "random_stations_graph.json"
    print("Reading network from file " + inputfile)
    with open(inputfile, "r") as file:
        data = json.loads(file.read())
        nodes = data["nodes"]
        edges = data["edges"]
    nw = Network(nodes, edges)

    def save_figure(filename, fig, debug=False, districts=True):
        print("Saving the network to file " + filename)
        nw.plot(fig=fig, debug=debug)
        nw.plot_edges(fig)
        nw.plot_nodes(fig, districts=districts)
        fig.savefig(filename, bbox_inches='tight', pad_inches=.0)

    fig = plt.figure(figsize=(19.20,10.87), dpi=129.05)
    save_figure("network.png",       fig, debug=False)
    fig = plt.figure(figsize=(19.20,10.80), dpi=1)
    save_figure("network.pdf",       fig, debug=False)
    fig = plt.figure(figsize=(19.20,10.80), dpi=1)
    save_figure("network_debug.pdf", fig, debug=True)
    fig = plt.figure(figsize=(19.20,10.80), dpi=1)
    save_figure("network_flattened.pdf", fig, debug=False, districts=False)
