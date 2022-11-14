import numpy as np
import networkx as nx
from timeline import Timeline
import matplotlib.pyplot as plt


class TimelineLoader(object):
    def __init__(self, filenames, mode):
        self.filenames = filenames
        self.mode = mode

    def load(self):
        timelines = {}

        if self.mode == "file":
            for i, filename in enumerate(self.filenames):
                timeline = Timeline()
                timeline_g = self.load_timeline(filename, timeline)
                _ = self.load_graph(timeline_g, timeline)
                timelines[i] = timeline
        else:
            timeline = Timeline()
            timeline_g = self.load_timeline(self.filenames, timeline)
            _ = self.load_graph(timeline_g, timeline)
            timelines[0] = timeline

        return timelines

    def load_timeline(self, graph_content, timeline):
        if self.mode == "file":
            data = open(graph_content, "r").read().splitlines()
        else:
            data = graph_content.split("\n")

        graphs, curr_graph, node_to_order, order_to_node, size = [], [], {}, {}, 0
        time = 0

        if self.mode == "string":
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

    def load_graph(self, graphs, timeline):
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
