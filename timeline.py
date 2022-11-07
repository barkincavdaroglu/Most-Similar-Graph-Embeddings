
class Timeline(object):
    def __init__(self):
        self.nx_graphs = {}
        self.node_to_order = {}
        self.order_to_node = {}
        self.size = {}
        self.time = 0
    
    def add_snapshpt(self, nx_graph, node_to_order, order_to_node, size):
        self.nx_graphs[self.time] = nx_graph
        self.node_to_order[self.time] = node_to_order
        self.order_to_node[self.time] = order_to_node
        self.size[self.time] = size
        self.time += 1

    def add_snapshot_graph_at_time(self, time, nx_graph):
        self.nx_graphs[time] = nx_graph

    def add_ordering_at_time(self, node_to_order, order_to_node, time):
        self.node_to_order[time] = node_to_order
        self.order_to_node[time] = order_to_node

    def add_size_at_time(self, time, size):
        self.size[time] = size

    def get_nx_graph_at_time(self, time):
        return self.nx_graphs[time]

    def get_matching_node(self, time, node_ord):
        return self.node_to_order[time][self.order_to_node[time][node_ord]]

    def get_all(self):
        for time in self.nx_graphs.keys():
            print(self.nx_graphs[time].edges(data=True))
            print(self.node_to_order[time])
            print(self.order_to_node[time])

    def get_snapshot_count(self):
        return len(self.nx_graphs.keys())