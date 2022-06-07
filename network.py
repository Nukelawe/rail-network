import matplotlib.pyplot as plt
import matplotlib
import json
import sys
import numpy as np
import matplotlib.transforms as transforms
from voronoi_chebyshev import voronoi_chebyshev

matplotlib.rc('font', family='serif', size=16)
matplotlib.rc('mathtext', fontset='cm')
matplotlib.rc('text', usetex=False)

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

# network style parameters
mec = "white" # color of lines
line_width = 7/6 # width of lines
bgcolor = "black" # background color
intersection_size = 8 # size of intersections
station_size = 7 # size of stations
# style arguments for lines
lineargs= {"linestyle":"solid", "marker":None, "color":mec, "linewidth":line_width}

class Network:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.stations = {}
        districts = {}
        for label in nodes.keys():
            if nodes[label]["station"]:
                self.stations[label] = self.nodes[label]
            # read districts from station labels
            addr = label.split(".")
            if len(addr) == 2 and addr[0].isnumeric():
                districts[label] = addr[0]
            else:
                districts[label] = 15
        self.districts = districts

        # dictionary that holds the number of connections to each node
        self.connections = {}
        for node in self.nodes:
            self.connections[node] = 0
        for edge in self.edges:
            self.connections[edge["from"]] += 1
            self.connections[edge["to"]] += 1
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

    def plot(self, fig, debug=False):
        artists = []
        ax = fig.gca()
        if debug:
            granularity = 6
            ax.set_xticks(np.arange(0, 192, granularity))
            ax.set_yticks(np.arange(0, 108, granularity))
            ax.grid()
        else:
            ax.axis("off")
        fig.set_facecolor(bgcolor)

        # plot edges
        for edge in self.edges:
            nodefrom = self.nodes[edge["from"]]
            nodeto = self.nodes[edge["to"]]
            edgetype = edge["type"]
            xmid = .5*(nodefrom["x"] + nodeto["x"])
            zmid = .5*(nodefrom["z"] + nodeto["z"])
            dx = nodeto["x"] - nodefrom["x"]; dx = 0
            dz = nodeto["z"] - nodefrom["z"]; dz = 0
            if edgetype == "zx":
                ax.plot([nodefrom["x"], nodefrom["x"], nodeto["x"]],
                        [ nodefrom["z"], nodeto["z"],   nodeto["z"]], **lineargs)
            elif edgetype == "xz":
                ax.plot([nodefrom["x"], nodeto["x"],   nodeto["x"]],
                        [ nodefrom["z"], nodefrom["z"], nodeto["z"]], **lineargs)
            elif edgetype == "xx":
                ax.plot([nodefrom["x"], xmid-.5*dz,    xmid+.5*dz,   nodeto["x"]],
                        [ nodefrom["z"], nodefrom["z"], nodeto["z"],  nodeto["z"]],
                        **lineargs)
            elif edgetype == "zz":
                ax.plot([nodefrom["x"], nodefrom["x"], nodeto["x"],  nodeto["x"]],
                        [ nodefrom["z"], zmid-.5*dx,    zmid+.5*dx,   nodeto["z"]],
                        **lineargs)

        # plot nodes
        trans = ax.transData
        trans_inv = ax.transData.inverted()
        offset = np.array([7.5/72., 7.5/72.])
        for label,node in self.nodes.items():
            color = banner_colors[district_colors[int(self.districts[label])]]
            if node["station"]:
                disp = trans.transform((node["x"],node["z"]))
                data = trans_inv.transform(disp)
                if node["intersection"]:
                    dataoffset = trans_inv.transform(disp + offset)
                    ax.plot([data[0], dataoffset[0]],
                            [data[1], dataoffset[1]], **lineargs)
                    data = dataoffset
                ax.plot(data[0], data[1], linestyle="none", markersize=station_size,
                        marker="o", markeredgewidth=line_width, markerfacecolor=bgcolor,
                        markeredgecolor=mec)
                # station labels
                ax.annotate(label, (data[0], data[1]), color=mec,
                        textcoords="offset pixels", xytext=(4,4),
                        fontname="sans serif", fontweight="regular", fontsize=16)
            if node["intersection"]:
                ax.plot(node["x"], node["z"], linestyle="none",
                        markersize=intersection_size, marker="D",
                        markeredgewidth=line_width, markerfacecolor=bgcolor,
                        markeredgecolor=mec)

        # color districts
        cells = self.district_boundaries()
        for district in cells:
            color = banner_colors[district_colors[int(district)]]
            patch = plt.Polygon(cells[district], linestyle=None, facecolor=color,
                    alpha=.25, edgecolor=bgcolor)
            ax.add_patch(patch)
        return artists

    # computes the voronoi diagram to visualize districts
    def district_boundaries(self):
        points = np.empty((len(self.stations), 2))
        districts = np.empty(len(self.stations))
        labels = []
        districts = {}
        for i,key in enumerate(self.stations.keys()):
            labels.append(key)
            points[i,0] = self.stations[key]["x"]
            points[i,1] = self.stations[key]["z"]
            if self.districts[key] not in districts.keys():
                districts[self.districts[key]] = []
            districts[self.districts[key]].append(i)
        xmin = np.amin(points[:,0])
        xmax = np.amax(points[:,0])
        zmin = np.amin(points[:,1])
        zmax = np.amax(points[:,1])
        xwidth = xmax - xmin
        zwidth = zmax - zmin
        r = 2
        cells = voronoi_chebyshev(points, districts,
                xmin-r*xwidth, xmax+r*xwidth,
                zmin-r*zwidth, zmax+r*zwidth)
        return cells

if __name__ =="__main__":
    if len(sys.argv) != 3:
        raise Exception("Incorrect number of arguments,\
                run as 'python network.py <inputfilename> <outputfilename>'")
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    print("Reading network from file " + inputfile)

    with open(inputfile, "r") as file:
        data = json.loads(file.read())
        nodes = data["nodes"]
        edges = data["edges"]

    network = Network(nodes, edges)
    dpi = 1
    fig = plt.figure(figsize=(1920/dpi,1080/dpi), dpi=dpi)
    ax = fig.gca()
    ax.set_xlim((0, 192))
    ax.set_ylim((0, 108))
    print("Saving the network to file " + outputfile)
    network.plot(fig)
    fig.savefig(outputfile, bbox_inches='tight', pad_inches=.0, dpi=dpi, backend="AGG",
            format="png")
