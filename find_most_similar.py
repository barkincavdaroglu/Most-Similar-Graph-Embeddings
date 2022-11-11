from load_data import load_all_timelines
import collections
import numpy as np
from n2vec import embed_graphs
from numpy.linalg import norm


def find_most_similar_timeline(prev_timeline_filenames, curr_timeline_filename, mode):
    timelines = load_all_timelines(prev_timeline_filenames, "file")

    current_timeline = load_all_timelines(curr_timeline_filename, mode)
    current_timeline = current_timeline[0]
    (
        current_snapshot_graph_0,
        current_snapshot_graph_1,
    ) = current_timeline.get_nx_graph_at_time(0), current_timeline.get_nx_graph_at_time(
        1
    )
    current_snapshot_model_0, current_snapshot_model_1 = embed_graphs(
        current_snapshot_graph_0
    ), embed_graphs(current_snapshot_graph_1)

    curr_min_euc, curr_min_pointer_euc = float("inf"), 0
    curr_max_cos, curr_max_pointer_cos = float("-inf"), 0

    model_embeds_of_timeline = collections.defaultdict(lambda: [])

    for key, timeline in timelines.items():  # key is timeline id
        for i in range(timeline.get_snapshot_count()):
            embedding_filename = (
                "timeline_" + str(key) + "_snapshot_" + str(i) + "_embedding"
            )
            embedding_model_filename = (
                "time_" + str(key) + "_" + str(i) + "_embedding_model"
            )
            edge_embedding_filename = (
                "time_" + str(key) + "_" + str(i) + "_edges_embedding"
            )
            model = embed_graphs(
                timeline.get_nx_graph_at_time(i),
                embedding_filename=embedding_filename,
                embedding_model_filename=embedding_model_filename,
                edges_embedding_filename=edge_embedding_filename,
            )
            model_embeds_of_timeline[key].append(model)

        for i in range(1, timeline.get_snapshot_count()):
            curr_prev_snapshot = model_embeds_of_timeline[key][i]
            prev_prev_snapshot = model_embeds_of_timeline[key][i - 1]

            node_vec1, node_vec2, node_vec3, node_vec4 = [], [], [], []

            for node in current_snapshot_graph_0.nodes():
                matched_order_in_other_timeline = timeline.get_matching_node(
                    i - 1, node
                )

                vec1 = current_snapshot_model_0.wv[node]
                node_vec1.append(np.array(vec1))

                vec3 = prev_prev_snapshot.wv[matched_order_in_other_timeline]
                node_vec3.append(np.array(vec1))

            for node in current_snapshot_graph_1.nodes():
                matched_order_in_other_timeline = timeline.get_matching_node(i, node)

                vec2 = current_snapshot_model_1.wv[node]
                node_vec2.append(np.array(vec1))

                vec4 = curr_prev_snapshot.wv[matched_order_in_other_timeline]
                node_vec4.append(np.array(vec2))

            node_vec1, node_vec2, node_vec3, node_vec4 = (
                np.array(node_vec1),
                np.array(node_vec2),
                np.array(node_vec3),
                np.array(node_vec4),
            )

            euc1 = np.linalg.norm(node_vec1 - node_vec3)
            euc2 = np.linalg.norm(node_vec2 - node_vec4)

            local_euc = euc1 + euc2

            cos1 = np.sum(
                np.dot(node_vec1, node_vec3.transpose())
                / (norm(node_vec1) * norm(node_vec3))
            )
            cos2 = np.sum(
                np.dot(node_vec2, node_vec4.transpose())
                / (norm(node_vec2) * norm(node_vec4))
            )

            local_cos = cos1 + cos2

            if local_euc < curr_min_euc:
                curr_min_euc = local_euc
                if euc1 < euc2:
                    curr_min_pointer_euc = (key, i)
                else:
                    curr_min_pointer_euc = (key, i - 1)

            if local_cos > curr_max_cos:
                curr_max_cos = local_cos
                if cos1 > cos2:
                    curr_max_pointer_cos = (key, i)
                else:
                    curr_max_pointer_cos = (key, i - 1)

    return curr_max_pointer_cos  # curr_min_pointer_euc#, curr_max_pointer_cos


prev_timeline_filenames = ["sample_data/1.txt", "sample_data/2.txt"]
curr_timeline_filename = ["sample_data/0.txt"]

"""print(
    find_most_similar_timeline(prev_timeline_filenames, curr_timeline_filename, "file")
)
print(
    find_most_similar_timeline(prev_timeline_filenames, curr_timeline_filename, "file")
)
print(
    find_most_similar_timeline(prev_timeline_filenames, curr_timeline_filename, "file")
)
print(
    find_most_similar_timeline(prev_timeline_filenames, curr_timeline_filename, "file")
)
print(
    find_most_similar_timeline(prev_timeline_filenames, curr_timeline_filename, "file")
)
print(
    find_most_similar_timeline(prev_timeline_filenames, curr_timeline_filename, "file")
)"""
