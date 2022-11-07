from load_data import load_all_timelines, load_timeline, load_graph
import collections   
import numpy as np
from n2vec import embed_graphs
from numpy.linalg import norm


def find_most_similar_timeline(filenames):
    timelines = load_all_timelines(filenames) 

    current_timeline = load_all_timelines(["sample_data/0.txt"])
    current_timeline = current_timeline[0]
    current_snapshot_graph = current_timeline.get_nx_graph_at_time(0)
    current_snapshot_model = embed_graphs(current_snapshot_graph)

    curr_min_euc, curr_min_pointer_euc = float('INF'), 0
    curr_max_cos, curr_max_pointer_cos = float('-INF'), 0

    model_embeds_of_timeline = collections.defaultdict(lambda: [])

    for key, timeline in timelines.items(): # key is timeline id
        for i in range(timeline.get_snapshot_count()):
            embedding_filename = "time_" + str(key) + "_" + str(i) + "_embedding"
            embedding_model_filename = "time_" + str(key) + "_" + str(i) + "_embedding_model"
            edge_embedding_filename = "time_" + str(key) + "_" + str(i) + "_edges_embedding"
            model = embed_graphs(timeline.get_nx_graph_at_time(i), embedding_filename=embedding_filename, embedding_model_filename=embedding_model_filename, edges_embedding_filename=edge_embedding_filename)
            model_embeds_of_timeline[key].append(model)
        
        for i in range(1, timeline.get_snapshot_count()):
            curr_prev_snapshot = model_embeds_of_timeline[key][i]
            prev_prev_snapshot = model_embeds_of_timeline[key][i - 1]

            for node in current_snapshot_graph.nodes():
                matched_order_in_other_timeline = timeline.get_matching_node(i, node)
                
                vec1 = current_snapshot_model.wv[node]
                vec2 = curr_prev_snapshot.wv[matched_order_in_other_timeline]
                dist1_euc = np.linalg.norm(vec1 - vec2)
                dist1_cos = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

                matched_order_in_other_timeline = timeline.get_matching_node(i - 1, node)

                vec3 = prev_prev_snapshot.wv[matched_order_in_other_timeline]
                dist2_euc = np.linalg.norm(vec1 - vec3)
                dist2_cos = np.dot(vec1, vec3)/(norm(vec1) * norm(vec3))

                if dist1_euc + dist2_euc < curr_min_euc:
                    curr_min_euc = dist1_euc + dist2_euc
                    if dist1_euc < dist2_euc:
                        curr_min_pointer_euc = (key, i)
                    else:
                        curr_min_pointer_euc = (key, i - 1)

                if dist1_cos + dist2_cos > curr_max_cos:
                    curr_max_cos = dist1_cos + dist2_cos
                    if dist1_cos > dist2_cos:
                        curr_max_pointer_cos = (key, i)
                    else:
                        curr_max_pointer_cos = (key, i - 1)
                        
    return curr_min_pointer_euc, curr_max_pointer_cos

    

filenames = ["sample_data/1.txt", "sample_data/2.txt"]

print(find_most_similar_timeline(filenames))
print(find_most_similar_timeline(filenames))
print(find_most_similar_timeline(filenames))
print(find_most_similar_timeline(filenames))
print(find_most_similar_timeline(filenames))
print(find_most_similar_timeline(filenames))