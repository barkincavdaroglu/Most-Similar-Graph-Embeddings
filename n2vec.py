import networkx as nx
from node2vec import Node2Vec
from numpy.random import seed

# Accepts a networkx graph and returns a its embedding using Node2Vec
def embed_graphs(
    graph,
    embedding_filename="",
    embedding_model_filename="",
    edges_embedding_filename="",
    embedding_dir="graph_embed_data/",
    save_embed_files_b=True,
):
    EMBEDDING_FILENAME = embedding_dir + embedding_filename + ".emb"

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(
        graph,
        dimensions=32,
        q=0.1,
        walk_length=30,
        num_walks=80,
        quiet=True,
        workers=8,  # , seed=seed(1)
    )

    # Embed nodes
    model = node2vec.fit(
        window=10, min_count=1, batch_words=4
    )  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Embed edges using Hadamard method
    from node2vec.edges import HadamardEmbedder

    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    if save_embed_files_b:
        model.wv.save_word2vec_format(EMBEDDING_FILENAME)

    return model
