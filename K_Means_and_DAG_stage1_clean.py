import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('combined_original.csv')

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Find the optimal number of clusters using silhouette method
def optimal_clusters(data, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    optimal_n_clusters = np.argmax(silhouette_scores) + 2  #(index 0 corresponds to 2 clusters)

    # Plot the silhouette scores for different cluster numbers
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    return optimal_n_clusters, silhouette_scores

# Find optimal number of clusters
max_clusters = 20
optimal_n_clusters, silhouette_scores = optimal_clusters(scaled_data, max_clusters)
print(f'Optimal number of clusters: {optimal_n_clusters}')

# Optimize n_init and max_iter parameters and track silhouette scores
def optimize_kmeans_params(data, n_clusters, n_init_values, max_iter_values):
    best_params = {}
    best_silhouette = -1
    silhouette_scores = []
    param_combinations = []

    for n_init in n_init_values:
        for max_iter in max_iter_values:
            kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)

            silhouette_scores.append(silhouette_avg)
            param_combinations.append((n_init, max_iter))

            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_params = {'n_init': n_init, 'max_iter': max_iter}

    return best_params, best_silhouette, silhouette_scores, param_combinations

# Possible values for n_init and max_iter
n_init_values = [10, 20, 30, 40]
max_iter_values = [100, 200, 300, 400, 500]

# Optimize KMeans parameters
best_params, best_silhouette, silhouette_scores, param_combinations = optimize_kmeans_params(
    scaled_data, optimal_n_clusters, n_init_values, max_iter_values
)

print(f'Best parameters: {best_params}')
# Apply KMeans with best parameters
kmeans = KMeans(n_clusters=optimal_n_clusters, n_init=best_params['n_init'], max_iter=best_params['max_iter'], random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_

# Use PCA to reduce dimensions
pca = PCA(2)
pca_data = pca.fit_transform(scaled_data)

# Plot clusters
plt.figure(figsize=(40, 20))
for i in range(optimal_n_clusters):
    plt.scatter(pca_data[labels == i, 0], pca_data[labels == i, 1], label=f'Cluster {i}')

# Label features inside clusters
for i, txt in enumerate(df.columns):
    plt.annotate(txt, (pca_data[i, 0], pca_data[i, 1]), fontsize = 24)

plt.title(f'K-Means Clustering Results with Two Clusters', fontsize=48)
plt.xlabel('PCA Component 1', fontsize=48)
plt.ylabel('PCA Component 2', fontsize=48)
plt.legend(fontsize=36)
plt.grid(True)
plt.savefig('K_Means_Clustering_Results.png', dpi=300)
plt.show()

print(f'Optimal number of clusters: {optimal_n_clusters}')
print(f'Best Parameters for KMeans: {best_params}')
print(f'Best Silhouette Score: {best_silhouette}')

# Plot individual clusters
for i in range(optimal_n_clusters):
    plt.figure(figsize=(60, 30))
    plt.scatter(pca_data[labels == i, 0], pca_data[labels == i, 1], label=f'Cluster {i}', c='C0')

    # Label features inside clusters
    for j, txt in enumerate(df.columns):
        plt.annotate(txt, (pca_data[j, 0], pca_data[j, 1]), fontsize=24)

    plt.title(f'Zoomed-in View of Cluster {i}', fontsize=48)
    plt.xlabel('PCA Component 1', fontsize=48)
    plt.ylabel('PCA Component 2', fontsize=48)
    plt.grid(True)
    plt.show()
# Apply KMeans with best parameters
kmeans = KMeans(
    n_clusters=optimal_n_clusters,
    n_init=best_params['n_init'],
    max_iter=best_params['max_iter'],
    random_state=42
)

kmeans.fit(scaled_data)
labels = kmeans.labels_

# Use PCA to reduce dimensions
pca = PCA(2)
pca_data = pca.fit_transform(scaled_data)

# Plot clusters
plt.figure(figsize=(40, 20))

for i in range(optimal_n_clusters):
    plt.scatter(
        pca_data[labels == i, 0],
        pca_data[labels == i, 1],
        label=f'Cluster {i}'
    )

# Label features inside clusters
for i, txt in enumerate(df.columns):
    plt.annotate(
        txt,
        (pca_data[i, 0], pca_data[i, 1]),
        fontsize=24   
    )

plt.title(
    f'K-Means Clustering Results with Two Clusters',
    fontsize=30       
)

plt.xlabel(
    'PCA Component 1',
    fontsize=24
)

plt.ylabel(
    'PCA Component 2',
    fontsize=24
)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.legend(fontsize=20)
plt.grid(True)

plt.savefig(
    'K_Means_Clustering_Results.png',
    dpi=300
)

plt.show()

print(f'Optimal number of clusters: {optimal_n_clusters}')
print(f'Best Parameters for KMeans: {best_params}')
print(f'Best Silhouette Score: {best_silhouette}')

# Plot individual clusters
for i in range(optimal_n_clusters):

    plt.figure(figsize=(60, 30))

    plt.scatter(
        pca_data[labels == i, 0],
        pca_data[labels == i, 1],
        label=f'Cluster {i}',
        c='C0'
    )

    # Label features inside clusters
    for j, txt in enumerate(df.columns):
        plt.annotate(
            txt,
            (pca_data[j, 0], pca_data[j, 1]),
            fontsize=36   # ← 再进一步放大
        )

    plt.title(
        f'Zoomed-in View of Cluster {i}',
        fontsize=40
    )

    plt.xlabel(
        'PCA Component 1',
        fontsize=28
    )

    plt.ylabel(
        'PCA Component 2',
        fontsize=28
    )

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.grid(True)

    plt.show()

import networkx as nx

 

G = nx.DiGraph()

 

environmental = [
    "Temperature",
    "PM2.5",
    "CO",
    "NO₂",
    "Ozone",
    "SO₂"
]

immune = [
    "HIV Status",
    "Inflammation",
    "Immune Senescence",
    "Activated CD8",
    "CD4 Depletion"
]

covariates = [
    "Age",
    "Race",
    "BMI",
    "Smoking",
    "Education"
]

outcomes = [
    "AAR",
    "EEAA",
    "PEAA",
    "GEAA",
    "DNAmTLadjAge"
]

 

for n in environmental:
    G.add_node(n, group="environment")

for n in immune:
    G.add_node(n, group="immune")

for n in covariates:
    G.add_node(n, group="covariate")

for n in outcomes:
    G.add_node(n, group="outcome")
 
 
# Environmental → Immune
 
for e in environmental:
    G.add_edge(e, "Inflammation")
    G.add_edge(e, "Immune Senescence")
    G.add_edge(e, "Activated CD8")
    G.add_edge(e, "CD4 Depletion")

 
# HIV → Immune
 
G.add_edge("HIV Status", "Inflammation")
G.add_edge("HIV Status", "Immune Senescence")
G.add_edge("HIV Status", "Activated CD8")
G.add_edge("HIV Status", "CD4 Depletion")

 
# Immune → Outcomes
 
for i in [
    "Inflammation",
    "Immune Senescence",
    "Activated CD8",
    "CD4 Depletion"
]:
    for o in outcomes:
        G.add_edge(i, o)

 
# Environmental → Outcomes
 
for e in environmental:
    for o in outcomes:
        G.add_edge(e, o)

 
# Covariates → Outcomes
 
for c in covariates:
    for o in outcomes:
        G.add_edge(c, o)

 
# Covariates → Immune
 
for c in covariates:
    G.add_edge(c, "Inflammation")
    G.add_edge(c, "Immune Senescence")
    G.add_edge(c, "Activated CD8")
    G.add_edge(c, "CD4 Depletion")

 

pos = {
    # Environmental
    "Temperature": (0, 5),
    "PM2.5": (0, 4),
    "CO": (0, 3),
    "NO₂": (0, 2),
    "Ozone": (0, 1),
    "SO₂": (0, 0),

    # HIV / Immune
    "HIV Status": (3, 4),
    "Inflammation": (6, 4),
    "Immune Senescence": (6, 2.8),
    "Activated CD8": (6, 1.6),
    "CD4 Depletion": (6, 0.4),

    # Covariates
    "Age": (3, -2),
    "Race": (4.8, -2),
    "BMI": (6.6, -2),
    "Smoking": (8.4, -2),
    "Education": (10.2, -2),

    # Outcomes
    "AAR": (14, 5.8),
    "EEAA": (14, 4),
    "PEAA": (14, 2.4),
    "GEAA": (14, 0.8),
    "DNAmTLadjAge": (14, -1),
}

 
# COLORS
 

node_colors = []

for node in G.nodes():
    group = G.nodes[node]["group"]

    if group == "environment":
        node_colors.append("#8ecae6")

    elif group == "immune":
        node_colors.append("#f28482")

    elif group == "covariate":
        node_colors.append("#90ee90")

    elif group == "outcome":
        node_colors.append("#d8a4e3")

 
# DRAW FIGURE
 

plt.figure(figsize=(18, 10), facecolor="#f7f7f7")
ax = plt.gca()
ax.set_facecolor("#f7f7f7")

# Edges
nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=16,
    edge_color="black",
    width=1.5,
    alpha=0.7,
    connectionstyle="arc3,rad=0.03"
)

# Nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=3200,
    edgecolors="black",
    linewidths=1.5
)

# Labels
nx.draw_networkx_labels(
    G,
    pos,
    font_size=10,
    font_weight="bold",
    font_family="sans-serif"
)

 
# LEGEND
 

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#8ecae6", edgecolor="black", label="Environmental Exposures"),
    Patch(facecolor="#f28482", edgecolor="black", label="HIV / Immune Pathways"),
    Patch(facecolor="#90ee90", edgecolor="black", label="Covariates"),
    Patch(facecolor="#d8a4e3", edgecolor="black", label="Epigenetic Aging Outcomes")
]

plt.legend(
    handles=legend_elements,
    loc="upper left",
    fontsize=11,
    frameon=True
)

 
# TITLE
 plt.title(
    "Directed Acyclic Graph (DAG): Environmental Exposures, Immune Dysregulation, and Epigenetic Aging",
    fontsize=18,
    fontweight="bold",
    pad=20
)

plt.axis("off")
plt.tight_layout()

 

plt.savefig(
    "publication_style_dag.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

G = nx.DiGraph()

environmental = [
    "Temperature",
    "PM2.5",
    "CO",
    "NO₂",
    "Ozone",
    "SO₂"
]

immune = [
    "HIV Status",
    "Inflammation",
    "Immune Senescence",
    "Activated CD8",
    "CD4 Depletion"
]

covariates = [
    "Age",
    "Race",
    "BMI",
    "Smoking",
    "Education"
]

outcomes = [
    "AAR",
    "EEAA",
    "PEAA",
    "GEAA",
    "DNAmTLadjAge"
]

 
for n in environmental:
    G.add_node(n, group="environment")

for n in immune:
    G.add_node(n, group="immune")

for n in covariates:
    G.add_node(n, group="covariate")

for n in outcomes:
    G.add_node(n, group="outcome")

 

 
for e in environmental:
    G.add_edge(e, "Inflammation")
    G.add_edge(e, "Immune Senescence")
    G.add_edge(e, "Activated CD8")
    G.add_edge(e, "CD4 Depletion")

 
G.add_edge("HIV Status", "Inflammation")
G.add_edge("HIV Status", "Immune Senescence")
G.add_edge("HIV Status", "Activated CD8")
G.add_edge("HIV Status", "CD4 Depletion")
 
for i in [
    "Inflammation",
    "Immune Senescence",
    "Activated CD8",
    "CD4 Depletion"
]:
    for o in outcomes:
        G.add_edge(i, o)

 
for e in environmental:
    for o in outcomes:
        G.add_edge(e, o)

 
for c in covariates:
    for o in outcomes:
        G.add_edge(c, o)

 
for c in covariates:
    G.add_edge(c, "Inflammation")
    G.add_edge(c, "Immune Senescence")
    G.add_edge(c, "Activated CD8")
    G.add_edge(c, "CD4 Depletion")

 

pos = {
    # Environmental
    "Temperature": (0, 5),
    "PM2.5": (0, 4),
    "CO": (0, 3),
    "NO₂": (0, 2),
    "Ozone": (0, 1),
    "SO₂": (0, 0),

    # HIV / Immune
    "HIV Status": (3, 4),
    "Inflammation": (6, 4),
    "Immune Senescence": (6, 2.8),
    "Activated CD8": (6, 1.6),
    "CD4 Depletion": (6, 0.4),

    # Covariates
    "Age": (3, -2),
    "Race": (4.8, -2),
    "BMI": (6.6, -2),
    "Smoking": (8.4, -2),
    "Education": (10.2, -2),

    # Outcomes
    "AAR": (14, 5.8),
    "EEAA": (14, 4),
    "PEAA": (14, 2.4),
    "GEAA": (14, 0.8),
    "DNAmTLadjAge": (14, -1),
}
 

node_colors = []

for node in G.nodes():
    group = G.nodes[node]["group"]

    if group == "environment":
        node_colors.append("#8ecae6")

    elif group == "immune":
        node_colors.append("#f28482")

    elif group == "covariate":
        node_colors.append("#90ee90")

    elif group == "outcome":
        node_colors.append("#d8a4e3")
 

plt.figure(figsize=(18, 10), facecolor="#f7f7f7")
ax = plt.gca()
ax.set_facecolor("#f7f7f7")

# Edges
nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=16,
    edge_color="black",
    width=1.5,
    alpha=0.7,
    connectionstyle="arc3,rad=0.03"
)

# Nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=3200,
    edgecolors="black",
    linewidths=1.5
)

# Labels
nx.draw_networkx_labels(
    G,
    pos,
    font_size=10,
    font_weight="bold",
    font_family="sans-serif"
)

 

legend_elements = [
    Patch(facecolor="#8ecae6", edgecolor="black", label="Environmental Exposures"),
    Patch(facecolor="#f28482", edgecolor="black", label="HIV / Immune Pathways"),
    Patch(facecolor="#90ee90", edgecolor="black", label="Covariates"),
    Patch(facecolor="#d8a4e3", edgecolor="black", label="Epigenetic Aging Outcomes")
]

plt.legend(
    handles=legend_elements,
    loc="upper left",
    fontsize=11,
    frameon=True
)
 

plt.title(
    "Directed Acyclic Graph (DAG): Environmental Exposures, Immune Dysregulation, and Epigenetic Aging",
    fontsize=18,
    fontweight="bold",
    pad=20
)

plt.axis("off")
plt.tight_layout()

 

plt.savefig(
    "publication_style_dag.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

G = nx.DiGraph()

environmental = ["Temperature", "PM2.5", "CO", "NO₂", "Ozone", "SO₂"]
immune = ["HIV Status", "Inflammation", "Immune Senescence", "Activated CD8", "CD4 Depletion"]
covariates = ["Age", "Race", "BMI", "Smoking", "Education"]
outcomes = ["AAR", "EEAA", "PEAA", "GEAA", "DNAmTLadjAge"]

for n in environmental:
    G.add_node(n, color="#5DADE2")

for n in immune:
    G.add_node(n, color="#EC7063")

for n in covariates:
    G.add_node(n, color="#58D68D")

for n in outcomes:
    G.add_node(n, color="#AF7AC5")

for e in environmental:
    G.add_edge(e, "Inflammation")
    G.add_edge(e, "Immune Senescence")

G.add_edge("HIV Status", "Inflammation")
G.add_edge("HIV Status", "Immune Senescence")
G.add_edge("HIV Status", "Activated CD8")
G.add_edge("HIV Status", "CD4 Depletion")

for i in immune[1:]:
    for o in outcomes:
        G.add_edge(i, o)

for c in covariates:
    for o in outcomes:
        G.add_edge(c, o)

pos = {
    "Temperature": (0, 5),
    "PM2.5": (0, 4),
    "CO": (0, 3),
    "NO₂": (0, 2),
    "Ozone": (0, 1),
    "SO₂": (0, 0),

    "HIV Status": (4, 5),
    "Inflammation": (7, 4),
    "Immune Senescence": (7, 3),
    "Activated CD8": (7, 2),
    "CD4 Depletion": (7, 1),

    "Age": (4, -1),
    "Race": (5.5, -1),
    "BMI": (7, -1),
    "Smoking": (8.5, -1),
    "Education": (10, -1),

    "AAR": (13, 5),
    "EEAA": (13, 4),
    "PEAA": (13, 3),
    "GEAA": (13, 2),
    "DNAmTLadjAge": (13, 1)
}

node_colors = [G.nodes[n]['color'] for n in G.nodes()]

plt.figure(figsize=(18, 10))

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowsize=18,
    width=1.5,
    alpha=0.45,
    edge_color="gray",
    connectionstyle="arc3,rad=0.04"
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=3400,
    node_color=node_colors,
    edgecolors="black",
    linewidths=1.5
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=10,
    font_weight="bold"
)

plt.title(
    "Conceptual DAG of Environmental Exposures, HIV-Related Immune Dysregulation, and Epigenetic Aging",
    fontsize=18,
    fontweight="bold"
)

plt.axis("off")
plt.tight_layout()
plt.show()

from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 8))

ax.set_xlim(0, 16)
ax.set_ylim(0, 10)

 

def add_box(x, y, text, color):
    box = FancyBboxPatch(
        (x, y),
        2.2,
        0.7,
        boxstyle="round,pad=0.03",
        linewidth=1.5,
        edgecolor="black",
        facecolor=color
    )
    ax.add_patch(box)
    ax.text(
        x + 1.1,
        y + 0.35,
        text,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold"
    )

# Environmental
add_box(0.5, 8, "Air Pollution", "#AED6F1")
add_box(0.5, 6.8, "Temperature", "#AED6F1")
add_box(0.5, 5.6, "Precipitation", "#AED6F1")

# HIV / Immune
add_box(5, 8, "HIV Status", "#F5B7B1")
add_box(5, 6.5, "Inflammation", "#F5B7B1")
add_box(5, 5, "Immune Senescence", "#F5B7B1")
add_box(5, 3.5, "CD8 Activation", "#F5B7B1")

# Covariates
add_box(5, 1, "Covariates", "#ABEBC6")

# Outcomes
add_box(11, 8, "AAR", "#D7BDE2")
add_box(11, 6.8, "EEAA", "#D7BDE2")
add_box(11, 5.6, "PEAA", "#D7BDE2")
add_box(11, 4.4, "GEAA", "#D7BDE2")
add_box(11, 3.2, "DNAmTLadjAge", "#D7BDE2")
 
# ARROWS
 
arrowprops = dict(arrowstyle="->", lw=1.5, color="black")

# Environment to immune
ax.annotate("", xy=(5, 6.9), xytext=(2.7, 8.2), arrowprops=arrowprops)
ax.annotate("", xy=(5, 5.4), xytext=(2.7, 6.9), arrowprops=arrowprops)

# HIV to immune
ax.annotate("", xy=(6.1, 6.9), xytext=(6.1, 8), arrowprops=arrowprops)
ax.annotate("", xy=(6.1, 5.4), xytext=(6.1, 6.5), arrowprops=arrowprops)

# Immune to outcomes
for y1, y2 in [(6.5,8),(6.5,6.8),(5,5.6),(5,4.4),(3.5,3.2)]:
    ax.annotate("", xy=(11, y2+0.35), xytext=(7.2, y1+0.35), arrowprops=arrowprops)

# Covariates
for y in [8,6.8,5.6,4.4,3.2]:
    ax.annotate("", xy=(11, y+0.35), xytext=(7.2,1.35), arrowprops=arrowprops)

plt.title(
    "Simplified Conceptual Pathways Linking HIV, Immune Dysregulation, and Epigenetic Aging",
    fontsize=17,
    fontweight="bold"
)

plt.axis("off")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(14, 8))

ax.set_xlim(0, 14)
ax.set_ylim(0, 10)

# Labels
ax.text(1, 8, "Environmental\nExposures", fontsize=15, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#85C1E9', edgecolor='black'))

ax.text(5, 8, "HIV Infection", fontsize=15, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#F1948A', edgecolor='black'))

ax.text(5, 5.5, "Immune Dysregulation\nInflammation\nImmune Senescence\nCD8 Activation",
        fontsize=13,
        bbox=dict(boxstyle='round', facecolor='#F5B7B1', edgecolor='black'))

ax.text(10, 7, "Epigenetic Aging Outcomes\nAAR\nEEAA\nPEAA\nGEAA\nDNAmTLadjAge",
        fontsize=13,
        bbox=dict(boxstyle='round', facecolor='#D2B4DE', edgecolor='black'))

ax.text(5, 2, "Covariates\nAge\nRace\nBMI\nSmoking\nEducation",
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='#A9DFBF', edgecolor='black'))

# Arrows
arrow = dict(arrowstyle='->', lw=2)

ax.annotate('', xy=(5,8), xytext=(2.5,8), arrowprops=arrow)
ax.annotate('', xy=(6.3,6.5), xytext=(6.3,8), arrowprops=arrow)
ax.annotate('', xy=(10,7), xytext=(7.8,6.5), arrowprops=arrow)
ax.annotate('', xy=(10,7), xytext=(6.5,2.8), arrowprops=arrow)
ax.annotate('', xy=(10,7), xytext=(2.5,8), arrowprops=arrow)

plt.title(
    "Proposed Causal Framework for Environmental Exposures and Epigenetic Aging",
    fontsize=18,
    fontweight='bold'
)

plt.axis('off')
plt.tight_layout()
plt.show()
 

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 1.2
})

 

G = nx.DiGraph()

environmental = [
    "Temperature",
    "PM2.5",
    "CO",
    "NO₂",
    "Ozone",
    "SO₂"
]

immune = [
    "Inflammation",
    "Immune\nSenescence",
    "Activated\nCD8",
    "CD4\nDepletion"
]

covariates = [
    "Age",
    "Race",
    "BMI",
    "Smoking",
    "Education"
]

outcomes = [
    "AAR",
    "EEAA",
    "PEAA",
    "GEAA",
    "DNAmTLadjAge"
]

 

for e in environmental:
    G.add_edge(e, "Inflammation")
    G.add_edge(e, "Immune\nSenescence")

G.add_edge("HIV", "Inflammation")
G.add_edge("HIV", "Immune\nSenescence")
G.add_edge("HIV", "Activated\nCD8")
G.add_edge("HIV", "CD4\nDepletion")

for i in immune:
    for o in outcomes:
        G.add_edge(i, o)

for c in covariates:
    for o in outcomes:
        G.add_edge(c, o)
 

pos = {

    # Environmental
    "Temperature": (0, 9),
    "PM2.5": (0, 7.5),
    "CO": (0, 6),
    "NO₂": (0, 4.5),
    "Ozone": (0, 3),
    "SO₂": (0, 1.5),

    # HIV
    "HIV": (3.5, 8),

    # Immune
    "Inflammation": (7, 8),
    "Immune\nSenescence": (7, 6),
    "Activated\nCD8": (7, 4),
    "CD4\nDepletion": (7, 2),

    # Covariates
    "Age": (3.5, -0.5),
    "Race": (5.2, -0.5),
    "BMI": (6.9, -0.5),
    "Smoking": (8.6, -0.5),
    "Education": (10.3, -0.5),

    # Outcomes
    "AAR": (13.5, 8.5),
    "EEAA": (13.5, 6.8),
    "PEAA": (13.5, 5.1),
    "GEAA": (13.5, 3.4),
    "DNAmTLadjAge": (13.5, 1.7)
}

 
colors = {}

for n in environmental:
    colors[n] = "#4C78A8"

colors["HIV"] = "#E45756"

for n in immune:
    colors[n] = "#F58518"

for n in covariates:
    colors[n] = "#54A24B"

for n in outcomes:
    colors[n] = "#B279A2"

 

fig, ax = plt.subplots(figsize=(18, 10))

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

 
# EDGES 

nx.draw_networkx_edges(
    G,
    pos,
    edge_color="#777777",
    width=1.6,
    alpha=0.45,
    arrows=True,
    arrowsize=18,
    arrowstyle="-|>",
    connectionstyle="arc3,rad=0.08"
)

 
# NODES
 

for node, (x, y) in pos.items():

    width = 1.9
    height = 0.72

    if node == "DNAmTLadjAge":
        width = 2.5

    patch = FancyBboxPatch(
        (x - width/2, y - height/2),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.5,
        edgecolor="black",
        facecolor=colors[node],
        alpha=0.96
    )

    ax.add_patch(patch)

    ax.text(
        x,
        y,
        node,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white"
    )

 
 

ax.text(
    -0.5,
    10.2,
    "Environmental Exposures",
    fontsize=15,
    fontweight="bold"
)

ax.text(
    5.2,
    10.2,
    "HIV / Immune Dysregulation",
    fontsize=15,
    fontweight="bold"
)

ax.text(
    13.2,
    10.2,
    "Epigenetic Aging Outcomes",
    fontsize=15,
    fontweight="bold",
    ha="center"
)

ax.text(
    6.5,
    -1.8,
    "Covariates",
    fontsize=14,
    fontweight="bold",
    ha="center"
)

 
 

plt.title(
    "Conceptual DAG Linking Environmental Exposures, HIV, and Epigenetic Aging",
    fontsize=20,
    fontweight="bold",
    pad=28
)

 

plt.xlim(-1.5, 16)
plt.ylim(-3, 11)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "nature_style_dag.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
