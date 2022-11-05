from load_data import load_timeline, load_graph
from embeds.n2vec import embed_graphs

graphs = load_timeline()
adj_matrices, nx_graphs = load_graph(graphs)

for i, nx_graph in enumerate(nx_graphs):
    embedding_filename = str(i) + "_embedding"
    embedding_model_filename = str(i) + "_embedding_model"
    edge_embedding_filename = str(i) + "_edges_embedding"
    embed_graphs(nx_graphs[i], embedding_filename=embedding_filename, embedding_model_filename=embedding_model_filename, edges_embedding_filename=edge_embedding_filename)