#

# class Node:
#     def wMean(self, x, y):
#         wm = np.sum((y / np.sum(y)) * x)
#         iwm = np.argmin(np.abs(wm - x))
#         return (wm, iwm)
#
#     def __init__(self, x, y, depth=4, minle=100):
#         # stop criteria parameters
#         self.len = len(x)
#         self.depth = depth
#
#         # stop criteria
#         if (len(x) < minle) | (depth < 1):
#             return
#
#         self.y = y
#         self.x = x
#         self.wmean, self.icent = self.wMean(self.x, self.y)
#
#         y_le = y[self.icent:]
#         x_le = x[self.icent:]
#
#         y_ri = y[:(self.icent-1)]
#         x_ri = x[:(self.icent-1)]
#
#         # recursion
#         self.left = Node(x=x_le, y=y_le, depth=depth-1, minle=minle)
#         self.right = Node(x=x_ri, y=y_ri, depth=depth-1, minle=minle)


class bst_search():
    # binary seach tree
    def __init__(self, x, y, depth=4, minle=100):
        import pandas as pd
        import numpy as np
        self.x = x
        self.y = y / np.max(y)
        self.depth = depth
        self.nData = []
        self.btree = self.Node(self.x, self.y, self.depth, minle)
        self.collect(self.btree)
        self.nodes = pd.DataFrame(self.nData)

    class Node:
        def __init__(self, x, y, depth=4, minle=1000):
            def wMean(x, y):
                import numpy as np
                wm = np.sum((y / np.sum(y)) * x)
                iwm = np.argmin(np.abs(wm - x))
                return (wm, iwm)

            # stop criteria parameters
            self.len = len(x)
            self.depth = depth

            # # stop criteria
            # if (len(x) < minle) | (depth < 1):
            #     return

            self.y = y
            self.x = x
            self.wmean, self.icent = wMean(self.x, self.y)

            y_le = y[self.icent:]
            x_le = x[self.icent:]

            y_ri = y[:(self.icent - 1)]
            x_ri = x[:(self.icent - 1)]

            # recursion
            if depth > 1:
                if len(y_le) > minle:
                    self.left = Node(x=x_le, y=y_le, depth=depth - 1, minle=minle)
                if len(y_ri) > minle:
                    self.right = Node(x=x_ri, y=y_ri, depth=depth - 1, minle=minle)

    def collect(self, btree, dir_par='root', c='0'):

        # traverse btree to collect weighted means
        # collect depth id, right or left and wmean
        # node ids coded as deptth + 0/1 for left/right, resp., so 0 (root), 10 (left child root), 11 (right child root), 100 (left child of left child root), 101 (right child of left child root, ...)
        nData = []

        if dir_par == 'root':
            nid = '0'
        elif dir_par =='l':
            nid = str(c) + str(0)
        elif dir_par == 'r':
            nid = str(c) + str(1)

        print(c)
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

    def vis_graph(self, xlab='Scantime (s)', ylab='Total Count', ax=None, col='green'):
        import networkx as nx

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
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        plt.vlines(vlines, ymin=0, ymax=y_min, color=col, linestyles='dotted')
        nx.draw(G, pos=xy, with_labels=False,
                node_size=3, node_color=col, font_size=6, \
                linewidths=3, node_shape='o', ax=ax, edge_color=col)
        limits = plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        return ax

class spec_distance():
    def __init__(self, specs, x, depth=4, minle=100, linkage='ward', **kwargs):
        import numpy as np
        self.linkage = linkage
        self.rr = []
        self.n = specs.shape[0]
        self.labels = np.array(['s'+str(i) for i in range(self.n)])
        for i in range(self.n):
            df = bst_search(x, specs[i], depth, minle).nodes
            wmean = df.wean
            wmean.index=df.id
            self.rr.append(wmean)
        self.wm = pd.concat(self.rr, axis=1)

        # calc distances

        D = np.zeros((self.n, self.n))
        for i in range(self.n-1):
            for j in range(i+1, self.n):
                D[i,j] = np.linalg.norm(self.wm.iloc[:,i].values-self.wm.iloc[:,j].values)

        self.D=np.maximum(D, d.transpose())

        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
        Z = hierarchy.linkage(pdist(self.D), self.linkage)

        plt.figure()
        dn = hierarchy.dendrogram(Z, labels=self.labels, **kwargs)
        plt.show()