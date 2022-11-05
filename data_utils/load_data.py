import numpy as np
import networkx as nx
from karateclub import DeepWalk, Walklets

def load_all_timelines(filenames):
    timelines = {}

    for i, filename in enumerate(filenames):
        timeline = load_timeline(filename)
        graphs_of_timeline = load_graph(timeline)
        timelines[i] = graphs_of_timeline

    return timelines

def load_timeline(filename="sample_data/1.txt"):
    temp = open(filename,'r').read().splitlines()
    graphs, curr_graph, node_to_order, size  = [], [], {}, 0

    for line in temp:
        if len(line) > 1:
            line_split = line.split()
            if line_split[0] == "O:":
                node_to_order[line_split[1]] = int(line_split[2])
                size += 1
            else:
                v1, v2, weight = node_to_order[line_split[1]], node_to_order[line_split[2]], float(line_split[3])
                curr_graph.append((v1, v2, weight))

        elif len(line) == 0: # another graph
            curr_graph.append(size)
            graphs.append(curr_graph)
            curr_graph = []
            size = 0

    curr_graph.append(size)
    graphs.append(curr_graph)

    return graphs

def load_graph(graphs):
    nx_graphs = []
    adj_matrices = []

    for graph in graphs:
        G = nx.Graph()

        graph_size = graph[-1]
        adj_matrix = np.zeros((graph_size, graph_size))
        
        for edge in graph[:-1]:
            v1, v2, weight = edge
            G.add_edge(v1, v2, weight=float(weight))

            adj_matrix[v1][v2] = weight
            adj_matrix[v2][v1] = weight
        adj_matrices.append(adj_matrix)
        nx_graphs.append(G)

    return adj_matrices, nx_graphs



"""model = DeepWalk()
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