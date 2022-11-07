import sys    
import collections   
import numpy as np
from load_data import load_all_timelines, load_timeline, load_graph
from n2vec import embed_graphs
from numpy.linalg import norm

sys.path.insert(1, '/Users/barkincavdaroglu/Desktop/Most-Similar-Graph-Embeddings/data_utils')     

def find_most_similar_timeline(filenames):
    timelines = load_all_timelines(filenames) # key: time, value: (adj_matrix, nx_graphs)

    current_snapshot = load_timeline("sample_data/0.txt")
    _, current_snapshot_graph = load_graph(current_snapshot)
    current_snapshot_graph = current_snapshot_graph[0]
    current_snapshot_model = embed_graphs(current_snapshot_graph)

    curr_min_euc, curr_min_pointer_euc = float('INF'), 0
    curr_min_cos, curr_min_pointer_cos = float('INF'), 0

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
                dist1_euc = np.linalg.norm(vec1 - vec2)
                dist1_cos = np.dot(vec1,vec2)/(norm(vec1) * norm(vec2))

                vec3 = prev_prev_snapshot.wv[node]
                dist2_euc = np.linalg.norm(vec1 - vec3)
                dist2_cos = np.dot(vec1,vec3)/(norm(vec1) * norm(vec3))

                if dist1_euc + dist2_euc < curr_min_euc:
                    curr_min_euc = dist1_euc + dist2_euc
                    if dist1_euc < dist2_euc:
                        curr_min_pointer_euc = (key, i)
                    else:
                        curr_min_pointer_euc = (key, i - 1)

                if dist1_cos + dist2_cos < curr_min_cos:
                    curr_min_cos = dist1_cos + dist2_cos
                    if dist1_cos < dist2_cos:
                        curr_min_pointer_cos = (key, i)
                    else:
                        curr_min_pointer_cos = (key, i - 1)
                        
    return curr_min_pointer_euc, curr_min_pointer_cos

filenames = ["sample_data/1.txt", "sample_data/2.txt"]

print(find_most_similar_timeline(filenames))
print(find_most_similar_timeline(filenames))
print(find_most_similar_timeline(filenames))
print(find_most_similar_timeline(filenames))
print(find_most_similar_timeline(filenames))
print(find_most_similar_timeline(filenames))

    
