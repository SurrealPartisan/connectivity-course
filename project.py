# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr

# Load:

dir = "./output/"

info = pd.read_csv(dir+"info.csv", usecols = ["SUB_ID", "DX_GROUP"])

nets = []
mats = []
infomap_clusters = []
infomap_clusters_grouped = []
for sub_id in info["SUB_ID"]:
    path = dir + str(sub_id) + ".net"
    net = nx.read_pajek(path)
    nets.append(net)
    mat = nx.adj_matrix(net).toarray()
    mats.append(mat)
    path2 = dir + "infomapout/" + str(sub_id) + ".clu"
    clu = np.loadtxt(path2, dtype=int, usecols=(0,1))
    infomap_clusters.append([n[1] for n in sorted([(n[0], n[1]) for n in clu])])
    infomap_clusters_grouped.append([])
    for i in range(1, max(clu[:, 1]) + 1):
        infomap_clusters_grouped[-1].append([n[0] for n in clu if n[1] == i])

mats = np.array(mats)

#%% Cluster:
    
lp = LabelPropagation()
x_all = []
y_all = []
x = []
y = []
repeats = 5
for i in range(len(nets)):
    x.append([])
    y.append([])
for seed_percentage in range(5, 96):
    print(repr(seed_percentage) + " %")
    for i in range(len(nets)):
        for j in range(repeats):
            n_nodes = len(infomap_clusters[i])
            labels = -np.ones(n_nodes)
            # Here I choose some "seed" nodes. Currently they are just random but 
            # perhaps they could be based on strength or clustering coefficient. 
            # Also would be good to ensure that each cluster is represented.
            seed_indices = np.random.choice(n_nodes, 
                                            size=n_nodes*seed_percentage//100, 
                                            replace=False)
            for n in seed_indices:
                labels[n] = infomap_clusters[i][n]
            # Fit:
            lp.fit(mats[i], labels)
            lp_clusters = np.array(lp.transduction_)
    
            # Compare:
            non_seed = np.setxor1d(range(n_nodes), seed_indices)
            rand = adjusted_rand_score(np.array(infomap_clusters[i])[non_seed],
                                       lp_clusters[non_seed])
            x_all.append(seed_percentage)
            x[i].append(seed_percentage)
            y_all.append(rand)
            y[i].append(rand)
x_all = np.array(x_all)
y_all = np.array(y_all)

#%% Visualize:

# Clustering similarity as a function of the amount of seeds:
fig = plt.figure()
fig.add_subplot(121)
for i in range(len(x)):
    plt.scatter(x[i], y[i])
    plt.show()
fig.add_subplot(122)
median = []
upper = []
lower = []
x_range = range(min(x_all), max(x_all) + 1)
for z in x_range:
    yset = y_all[np.where(x_all == z)[0]]
    median.append(np.median(yset))
    upper.append(np.percentile(yset, 95))
    lower.append(np.percentile(yset, 5))
plt.plot(x_range, median, 'k')
plt.plot(x_range, upper, 'k--')
plt.plot(x_range, lower, 'k--')

# A sample network:
i = np.random.choice(range(len(nets)))
net = nets[i]

# Fit once more for the fig
seed_percentage = 10
n_nodes = len(infomap_clusters[i])
labels = -np.ones(n_nodes)
# Here I choose some "seed" nodes. Currently they are just random but 
# perhaps they could be based on strength or clustering coefficient. 
# Also would be good to ensure that each cluster is represented.
seed_indices = np.random.choice(n_nodes, 
                                size=n_nodes*seed_percentage//100, 
                                replace=False)
for n in seed_indices:
    labels[n] = infomap_clusters[i][n]
# Fit:
lp.fit(mats[i], labels)
lp_clusters = np.array(lp.transduction_)

weights = np.array([data['weight'] for (i, j, data) in net.edges(data=True)])
pos=nx.spring_layout(net) # positions for all nodes
n_clusters = len(infomap_clusters_grouped[i])
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
node_color = [colors[c-1] for c in infomap_clusters[i]]
fig = plt.figure()
ax = fig.add_subplot(121)
nx.draw_networkx_nodes(net, ax=ax, pos=pos, node_size=100, node_color=node_color)
nx.draw_networkx_edges(net, width=weights, ax=ax, pos=pos)
ax = fig.add_subplot(122)
node_color = [colors[int(c)-1] for c in lp_clusters]
nx.draw_networkx_nodes(net, ax=ax, pos=pos, node_size=100, node_color=node_color)
nx.draw_networkx_edges(net, width=weights, ax=ax, pos=pos)
plt.show()

#%% Analyze:

corr, p = spearmanr(x_all, y_all)
print("The Spearman rank correlation coefficient for clustering similarity")
print("and seed percentage is " + repr(corr) + " with p-value " + repr(p) + ".")

# Let's bootstrap some confidence intervals

boots = []
for i in range(5000):
    inds = np.random.choice(len(x_all), len(x_all))
    r, _ = spearmanr(x_all[inds], y_all[inds])
    boots.append(r)
rup = np.percentile(boots, 97.5)
rlo = np.percentile(boots, 2.5)
print("The 95 % confidence interval is (" + repr(rlo) + "..." + repr(rup) + ".")