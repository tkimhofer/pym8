# Hierarchical clustering based on a binary tree similarity
import matplotlib.pyplot as plt
cc = [0,1,2,3,5,6,7,8,9,10,11, 12,13, 14, 15]
Xn=Xn[cc]
f, x = plt.subplots(1, 1)
c=0
for i in range(Xn.shape[0]):
    plotting.spec((Xn[[i], :]/np.max(Xn[i]))+(c*0.3), ppe, c='black', ax=[f,x], shift=[0, 9])
    plt.annotate('s'+str(c), (9, (c*0.3+0.05)))
    c += 1


t1 = BstSearch(ppe, Xn[0]/np.max(Xn[0]))
ax = t1.vis_graph(xlab='ppm', ylab='Int', ax=None, col='cyan', invertx=True)
t2 = BstSearch(ppe, Xn[1]/np.max(Xn[1]))
t2.vis_graph(xlab='ppm', ylab='Int', ax=ax, col='magenta', invertx=False)

# aa = spec_distance(Xn, ppe, metric='euclidean', linkage='ward', minle=4, color_threshold=10, above_threshold_color='black')

cl1 = [13, 5, 11, 7, 1,2,10]
cl2 = [8, 14, 0, 6]
cl3 = [4, 12, 3, 9]

cl=cl2+cl3+cl1
f, x = plt.subplots(1, 1)
c=0
for i in cl:
    if i in cl1:
        col ='orange'
    elif i in cl2:
        col ='green'
    else:
        col='red'
    plotting.spec((Xn[[i], :]/np.max(Xn[i]))+(c*0.3), ppe, c=col, ax=[f,x], shift=[0, 9])
    plt.annotate('s'+str(i), (9, (c*0.3+0.05)))
    c += 1

a=plotting.spec(Xn[cl2], ppe)

fig, axs = plt.subplots(3,1)
plotting.spec(Xn[cl1], ppe, ax=[fig, axs[0]], c='orange')
plotting.spec(Xn[cl2], ppe, ax=[fig, axs[1]], c='lightgreen')
plotting.spec(Xn[cl3], ppe, ax=[fig, axs[2]], c='darkgreen')

Xs=Xn / np.max(Xn[:, np.where((ppe>3 ) &(ppe<3.1))[0]], 1)[..., np.newaxis]


axs[0] = a[1]
axs[1] = plotting.spec(Xn[cl2], ppe)[1]

axs[2] = plotting.spec(Xn[cl3], ppe)[1]



class Node:
    # defines cutpoints based on y-weighted mean of x values
    def wMean(self, x, y):
        import numpy as np
        wm = np.sum((y / np.sum(y)) * x)
        iwm = np.argmin(np.abs(wm - x))
        return (wm, iwm)

    def __init__(self, x, y, depth=4, minle=100):
        self.len = len(x)
        self.depth = depth

        # check stopping criteria
        if (len(x) < minle) | (depth < 1):
            return

        self.y = y
        self.x = x
        self.wmean, self.icent = self.wMean(self.x, self.y)

        y_le = y[self.icent:]
        x_le = x[self.icent:]

        y_ri = y[:(self.icent-1)]
        x_ri = x[:(self.icent-1)]

        # recursion
        self.left = Node(x=x_le, y=y_le, depth=depth-1, minle=minle)
        self.right = Node(x=x_ri, y=y_ri, depth=depth-1, minle=minle)

class BstSearch():
    # calculates binary tree (btree) for a single spectrum
    def __init__(self, x, y, depth=4, minle=100):
        # depth: recursion depth to define x-cutpoints (recursion stopping criterion)
        # minle: minimum length of a segment defined by right and left cutpoint (recursion stopping criterion)
        import pandas as pd
        import numpy as np
        self.x = x
        self.y = y / np.max(y)
        self.depth = depth
        self.nData = []
        self.btree = Node(self.x, self.y, self.depth, minle)
        self.collect(self.btree)
        self.nodes = pd.DataFrame(self.nData)

    # class Node:
    #     def callNode(self, x, y, depth, minle):
    #         self.Node(x=x, y=y, depth=depth - 1, minle=minle)
    #
    #     def __init__(self, x, y, depth=4, minle=1000):
    #         import numpy as np
    #
    #         def wMean(x, y):
    #             wm = np.sum((y / np.sum(y)) * x)
    #             iwm = np.argmin(np.abs(wm - x))
    #             return (wm, iwm)
    #
    #         # stop criteria parameters
    #         self.len = len(x)
    #         self.depth = depth
    #
    #         self.y = y
    #         self.x = x
    #         self.wmean, self.icent = wMean(self.x, self.y)
    #
    #         y_le = y[self.icent:]
    #         x_le = x[self.icent:]
    #
    #         y_ri = y[:(self.icent - 1)]
    #         x_ri = x[:(self.icent - 1)]
    #
    #         # recursion
    #         if depth > 1:
    #             if len(y_le) > minle:
    #                 self.left = self.callNode(x=x_le, y=y_le, depth=depth - 1, minle=minle)
    #             if len(y_ri) > minle:
    #                 self.right = self.callNode(x=x_ri, y=y_ri, depth=depth - 1, minle=minle)
    def collect(self, btree, dir_par='root', c='0'):
        # traverse btree to collect weighted means
        # collect depth value, right or left and wmean (cutpoints)
        nData = []

        if dir_par == 'root':
            nid = '0'
        elif dir_par =='l':
            nid = str(c) + str(0)
        elif dir_par == 'r':
            nid = str(c) + str(1)

        if hasattr(btree, 'wmean'):
            data = {'id': nid, 'wean': btree.wmean}
            self.nData.append(data)
            nData.append(data)

        if hasattr(btree, 'left'):
            nData = nData + [self.collect(btree.left, dir_par='l', c=nid)]
        else:
            return

        if hasattr(btree, 'right'):
            nData = nData + [self.collect(btree.right, dir_par='r', c=nid)]
        else:
            return

        return nData

    def vis_graph(self, xlab='Scantime (s)', ylab='Total Count', ax=None, col='green', invertx=False):
        #  binary tree visualised over input data
        import networkx as nx
        import numpy as np
        import matplotlib.pyplot as plt

        def srange(t, tdiff, tmin, x_diff, x_min):
            return ((x_diff * (t - tmin)) / tdiff) + x_min
        y_max = np.max(self.y) * 2
        y_min = np.max(self.y) * 0.8
        y_diff = y_max - y_min

        tmax=np.log(self.depth+1)

        A = np.zeros((len(self.nodes), len(self.nodes)))
        N = self.nodes.id.values
        U = self.nodes[1:].id.values
        xy = {'0': np.array([self.nodes.wean[0], srange(t=np.log(self.depth + 1), tdiff=tmax, tmin=0, x_diff=y_diff, x_min=y_min)])}
        adj_list = []
        c = 1
        while len(U) > 0:
            ix = np.where(N == U[0][0:-1])[0][0]
            adj_list.append((N[ix], U[0]))
            A[ix, c] = 1
            xy.update({U[0]: np.array([self.nodes.wean.iloc[c], srange(t=np.log((self.depth+1)-len(U[0])), tdiff=tmax, tmin=0, x_diff=y_diff, x_min=y_min) ])})
            c=c+1
            U = U[1:]

        # adjacency order is test.nodes.id
        self.tips = (np.sum(np.maximum(A, A.transpose()), 0) == 1.)

        #vlines=np.array([(xy[x][0], 0, xy[x][1]) for x in self.nodes.id.iloc[np.array(self.tips)]])
        vlines = np.array([xy[x][0] for x in self.nodes.id.iloc[np.array(self.tips)]])

        G = nx.from_numpy_matrix(A)
        nmap = dict(zip(sorted(G), self.nodes.id.values))
        G = nx.relabel_nodes(G, nmap)

        if isinstance(ax, type(None)):
            fig, ax = plt.subplots()
        ax.plot(self.x, self.y, color=col, linewidth=0.7)
        if invertx:
            ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        plt.vlines(vlines, ymin=0, ymax=y_min, color=col, linestyles='dotted')
        nx.draw(G, pos=xy, with_labels=False,
                node_size=3, node_color=col, font_size=6, \
                linewidths=3, node_shape='o', ax=ax, edge_color=col)
        limits = plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        return ax

class SpecDistance():
    # hierarchical clustering based on binary tree similarity
    def __init__(self, specs, x, depth=4, minle=4, metric='euclidean', linkage='ward', **kwargs):
        # specs: input data with rows representing individual spectra
        # x: x variables, len(x) == specs.shape[1]
        # depth: recursion depth to define x-axis cutpoints (recursion stopping criterion)
        # minle: minimum length of a segment defined by right and left cutpoint (recursion stopping criterion)
        # metric: distance metric (see scipy.spatial.distance, most useful ones are ‘cosine’ and ‘euclidean’)
        # linkage: linkage function for hierarchical agglomerative clustering
        # **kwargs: additional parameters for scipy function dendrogram (hierarchy module)

        # returns cluster dendrogram
        import numpy as np
        import pandas as pd
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
        import matplotlib.pyplot as plt

        self.linkage = linkage
        self.rr = []
        self.n = specs.shape[0]
        self.labels = np.array(['s'+str(i) for i in range(self.n)])
        for i in range(self.n):
            df = BstSearch(x, specs[i], depth, minle).nodes
            wmean = df.wean
            wmean.index=df.id
            self.rr.append(wmean)
        self.wm = pd.concat(self.rr, axis=1)

        D = np.zeros((self.n, self.n))
        for i in range(self.n-1):
            for j in range(i+1, self.n):
                D[i,j] = np.linalg.norm(self.wm.iloc[:,i].values-self.wm.iloc[:,j].values)

        self.D=np.maximum(D, D.transpose())

        Z = hierarchy.linkage(pdist(self.D, metric=metric), self.linkage)

        plt.figure()
        hierarchy.dendrogram(Z, labels=self.labels, **kwargs)
        plt.show()
