# We will take a snapshot of a graph and find the sequence of graphs most similar to it

# importing the sys module
import sys    
import collections   
import numpy as np

sys.path.insert(1, '/Users/barkincavdaroglu/Desktop/Most-Similar-Graph-Embeddings/data_utils')     

from load_data import load_all_timelines, load_timeline, load_graph
from n2vec import embed_graphs

def find_most_similar_timeline(filenames=["sample_data/1.txt", "sample_data/2.txt"]):
    timelines = load_all_timelines(filenames) # key: time, value: (adj_matrix, nx_graphs)

    current_snapshot = load_timeline("sample_data/0.txt")
    _, current_snapshot_graph = load_graph(current_snapshot)
    current_snapshot_graph = current_snapshot_graph[0]
    current_snapshot_model = embed_graphs(current_snapshot_graph)

    curr_min, curr_min_pointer = float('INF'), 0
    model_embeds_of_timeline = collections.defaultdict(lambda: [])

    for key, val in timelines.items(): # key is timeline id
        for i, nx_graph in enumerate(val[1]):
            embedding_filename = "time_" + str(key) + "_" + str(i) + "_embedding"
            embedding_model_filename = "time_" + str(key) + "_" + str(i) + "_embedding_model"
            edge_embedding_filename = "time_" + str(key) + "_" + str(i) + "_edges_embedding"
            model = embed_graphs(val[1][i], embedding_filename=embedding_filename, embedding_model_filename=embedding_model_filename, edges_embedding_filename=edge_embedding_filename)
            model_embeds_of_timeline[key].append(model)
        
        for i in range(1, len(val[1])):
            curr_prev_snapshot = model_embeds_of_timeline[key][i]
            prev_prev_snapshot = model_embeds_of_timeline[key][i - 1]

            for node in current_snapshot_graph.nodes():
                vec1 = current_snapshot_model.wv[node]
                vec2 = curr_prev_snapshot.wv[node]
                dist1 = np.linalg.norm(vec1 - vec2)

                vec3 = prev_prev_snapshot.wv[node]
                dist2 = np.linalg.norm(vec1 - vec3)
                if dist1 + dist2 < curr_min:
                    curr_min = dist1 + dist2
                    if dist1 < dist2:
                        curr_min_pointer = (key, i)
                    else:
                        curr_min_pointer = (key, i - 1)

    return curr_min_pointer

print(find_most_similar_timeline())
    

    
