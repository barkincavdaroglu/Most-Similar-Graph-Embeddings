import math
import numpy as np
import networkx as nx
from karateclub import DeepWalk, Walklets
from timeline import Timeline
from n2vec import embed_graphs
import os
import matplotlib.pyplot as plt


def load_all_timelines(filenames, mode):
    timelines = {}

    if mode == "file":
        for i, filename in enumerate(filenames):
            timeline = Timeline()
            timeline_g = load_timeline(filename, timeline, mode)
            _ = load_graph(timeline_g, timeline)
            timelines[i] = timeline
    else:
        timeline = Timeline()
        timeline_g = load_timeline(filenames, timeline, mode)
        _ = load_graph(timeline_g, timeline)
        timelines[0] = timeline

    return timelines


def load_timeline(graph_content, timeline, mode):
    if mode == "file":
        data = open(graph_content, "r").read().splitlines()
    else:
        data = graph_content.split("\n")

    graphs, curr_graph, node_to_order, order_to_node, size = [], [], {}, {}, 0
    time = 0

    if mode == "string":
        for line in data:
            print(line)

    for line in data:
        if len(line) > 1:
            line_split = line.split()
            if line_split[0] == "O:":
                node_to_order[line_split[1]] = int(line_split[2])
                order_to_node[int(line_split[2])] = line_split[1]
                size += 1
            else:
                v1, v2, weight = (
                    node_to_order[line_split[1]],
                    node_to_order[line_split[2]],
                    float(line_split[3]),
                )
                curr_graph.append((v1, v2, weight))

        elif len(line) == 0:  # another graph
            timeline.add_ordering_at_time(node_to_order, order_to_node, time)
            curr_graph.append(size)
            graphs.append(curr_graph)
            curr_graph = []
            node_to_order = {}
            order_to_node = {}
            time += 1
            size = 0

    timeline.add_ordering_at_time(node_to_order, order_to_node, time)
    curr_graph.append(size)
    graphs.append(curr_graph)

    return graphs


def load_graph(graphs, timeline):
    adj_matrices = []
    time = 0

    for graph in graphs:
        G = nx.Graph()

        graph_size = graph[-1]
        timeline.add_size_at_time(time, graph_size)
        adj_matrix = np.zeros((graph_size, graph_size))

        for edge in graph[:-1]:
            v1, v2, weight = edge
            G.add_edge(v1, v2, weight=(float(weight) * 100) ** 2)

            adj_matrix[v1][v2] = weight
            adj_matrix[v2][v1] = weight
        adj_matrices.append(adj_matrix)
        timeline.add_snapshot_graph_at_time(time, G)
        time += 1

    return adj_matrices


"""
filenames = []
for filename in os.listdir("timelines"):
    f = os.path.join("timelines", filename)
    # checking if it is a file
    if os.path.isfile(f):
        filenames.append(f)


timelines = load_all_timelines(filenames, "file")

for i in timelines.keys():
    current_snapshot_graph = timelines[i].get_nx_graph_at_time(0)
    embed_graphs(current_snapshot_graph, str(i))

# filenames = ["sample_data/1.txt", "sample_data/2.txt"]
filenames = ["sample_data/0.txt"]
timelines = load_all_timelines(filenames, "file")
nx_graphs = []
for i in timelines.keys():
    for j in range(timelines[i].get_snapshot_count()):
        nx_graphs.append(timelines[i].get_nx_graph_at_time(j))

for G in nx_graphs:
    # change the weights of edges
    for u, v, d in G.edges(data=True):
        d["weight"] = math.sqrt(d["weight"]) / 100

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

    pos = nx.spring_layout(
        G, k=0.18, iterations=20
    )  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


model = DeepWalk()
model.fit(nx_graphs[0])
embedding = model.get_embedding()
print("EMBEDDING: ", embedding, "\n")

model = Walklets()
model.fit(nx_graphs[0])
embedding = model.get_embedding()
print("EMBEDDING: ", embedding)

# read the embedding file
def read_embedding(json_dicts, id_to_int_mapping, filename="embeddings.emb"):
    with open(filename, "r") as f:
        lines = f.readlines()

        num_nodes, dim = lines[0].split()
        num_nodes, dim = int(num_nodes), int(dim)
        node_embeddings = np.zeros((num_nodes, dim))
        for i in range(1, len(lines)):
            line = lines[i].split()
            node_id = line[0]
            embedding_str = line[1:]
            str_fts = ""
            for ft in embedding_str:
                str_fts += ft + " "
            json_dicts[0]["features"][id_to_int_mapping[node_id]] = str_fts

            embedding = [float(x) for x in line[1:]]
            node_embeddings[id_to_int_mapping[node_id]] = embedding
    return node_embeddings

node_embeddings = read_embedding(json_dicts=json_dicts, id_to_int_mapping=id_to_int_mapping)

print(json_dicts)

# Serializing json
json_object = json.dumps(json_dicts[0], indent=4)
 
# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)"""
