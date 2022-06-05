import matplotlib.pyplot as plt
import matplotlib
import json
import sys
import numpy as np
import matplotlib.transforms as transforms
#from scipy.spatial import Voronoi
#from voronoiz import voronoi_l1

matplotlib.rc('font', family='serif', size=16)
matplotlib.rc('mathtext', fontset='cm')
matplotlib.rc('text', usetex=True)

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

class Network:
    def __init__(self, nodes, edges, plot_districts=False):
        self.nodes = nodes
        self.edges = edges
        self.stations = {}
        districts = {}
        for label in nodes.keys():
            if nodes[label]["station"]:
                self.stations[label] = self.nodes[label]
            if plot_districts: # attempt to categorize by label
                addr = label.split(".")
                if len(addr) == 2:
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

    def plot(self, filename, colors=None):
        print("Saving the network to file " + filename)
        fig = plt.figure(figsize=(19.20,10.80), dpi=1)
        #fig.set_facecolor("black")
        ax = plt.gca()
        #ax.axis("off")
        ax.set_aspect('equal', adjustable='box')
        granularity = 6
        plt.xticks(np.arange(0, 192, granularity))
        plt.yticks(np.arange(0, 108, granularity))
        plt.grid()
        mec = "black"

        # plot edges
        for edge in self.edges:
            nodefrom = self.nodes[edge["from"]]
            nodeto = self.nodes[edge["to"]]
            edgetype = edge["type"]
            xmid = .5*(nodefrom["x"] + nodeto["x"])
            zmid = .5*(nodefrom["z"] + nodeto["z"])
            kwargs = {"linestyle":"solid", "marker":None, "color":mec, "linewidth":8/6}
            if edgetype == "zx":
                plt.plot([nodefrom["x"], nodefrom["x"], nodeto["x"]],
                        [ nodefrom["z"], nodeto["z"],   nodeto["z"]], **kwargs)
            elif edgetype == "xz":
                plt.plot([nodefrom["x"], nodeto["x"],   nodeto["x"]],
                        [ nodefrom["z"], nodefrom["z"], nodeto["z"]], **kwargs)
            elif edgetype == "xx":
                plt.plot([nodefrom["x"], xmid,          xmid,         nodeto["x"]],
                        [ nodefrom["z"], nodefrom["z"], nodeto["z"],  nodeto["z"]],
                        **kwargs)
            elif edgetype == "zz":
                plt.plot([nodefrom["x"], nodefrom["x"], nodeto["x"],  nodeto["x"]],
                        [ nodefrom["z"], zmid,          zmid,         nodeto["z"]],
                        **kwargs)

        # plot nodes
        offset = transforms.ScaledTranslation(-5.5/72., -5.5/72., plt.gcf().dpi_scale_trans)
        trans = plt.gca().transData
        for label,node in self.nodes.items():
            color = banner_colors[district_colors[int(self.districts[label])]]
            plt.annotate(label, (node["x"], node["z"]), color="white",
                    textcoords="offset pixels", xytext=(4,4))
            if node["intersection"]:
                plt.plot(node["x"], node["z"], linestyle="none", markersize=8,
                        marker="D", markeredgewidth=7/6, markerfacecolor=color,
                        markeredgecolor=mec)
                if node["station"]:
                    plt.plot(node["x"], node["z"], linestyle="none", markersize=7,
                            marker="o", markeredgewidth=7/6, markerfacecolor=color,
                            markeredgecolor=mec, transform=trans+offset)
            elif node["station"]:
                plt.plot(node["x"], node["z"], linestyle="none", markersize=7,
                        marker="o", markeredgewidth=7/6, markerfacecolor=color,
                        markeredgecolor=mec)
            else:
                raise Exception("A node must be either intersection, station or both")

        self.district_boundaries()
        plt.savefig(filename, bbox_inches='tight', pad_inches=.0)

    # computes the voronoi diagram to visualize districts
    def district_boundaries(self):
        pass
        #return self.voronoi_manhattan()
        #return self.voronoi_euclidean()

    def voronoi_euclidean(self):
        points = np.empty((len(self.stations), 2))
        labels = []
        for i,key in enumerate(self.stations.keys()):
            labels.append(key)
            points[i,0] = self.stations[key]["x"]
            points[i,1] = self.stations[key]["z"]
        voronoi = Voronoi(points)
        ver = voronoi.vertices
        rp = voronoi.ridge_points
        for i,ind in enumerate(voronoi.ridge_vertices):
            print(labels[rp[i,0]].split(".")[0] + " = " + labels[rp[i,1]].split(".")[0])
            if labels[rp[i,0]].split(".")[0] != labels[rp[i,1]].split(".")[0]:
                plt.plot(ver[ind,0], ver[ind,1], linestyle="dashed", linewidth=1,
                        marker=None, markeredgewidth=8/6, color="white")

    def voronoi_manhattan(self):
        points = np.empty((len(self.stations), 2))
        labels = []
        for i,key in enumerate(self.stations.keys()):
            labels.append(key)
            points[i,0] = self.stations[key]["x"]
            points[i,1] = self.stations[key]["z"]
        xmin = np.amin(points[:,0])
        xmax = np.amax(points[:,0])
        zmin = np.amin(points[:,1])
        zmax = np.amax(points[:,1])
        xwidth = xmax - xmin
        zwidth = zmax - zmin
        r = .1
        cells = voronoi_l1(points,
                xmin-r*xwidth, xmax+r*xwidth,
                zmin-r*zwidth, zmax+r*zwidth)
        for cell in cells:
            plt.plot(cell[:,0], cell[:,1], linestyle="dashed", linewidth=1,
                    marker=None, markeredgewidth=8/6, color="white")

def intersection_key(x,z):
    return str(x)+","+str(z)

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

    network = Network(nodes, edges, plot_districts=True)
    network.plot(outputfile)
