import networkx as nx
from node2vec import Node2Vec
from numpy.random import seed

# Accepts a networkx graph and returns a its embedding using Node2Vec
def embed_graphs(graph, embedding_filename="", embedding_model_filename="", edges_embedding_filename="", embedding_dir="graph_embed_data/"):
    EMBEDDING_FILENAME = embedding_dir + embedding_filename + ".emb" # "graph_embed_data/embeddings.emb"
    EMBEDDING_MODEL_FILENAME = embedding_dir + embedding_model_filename + ".model" # "graph_embed_data/embeddings.model"
    EDGES_EMBEDDING_FILENAME = embedding_dir + edges_embedding_filename + ".emb" # "graph_embed_data/edges.emb"

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(
        graph, dimensions=32, walk_length=20, num_walks=80, workers=1, seed=seed(1)
    ) 

    # Embed nodes
    model = node2vec.fit(
        window=10, min_count=1, batch_words=4
    )  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Save embeddings for later use
    #model.wv.save_word2vec_format(EMBEDDING_FILENAME)

    # Save model for later use
    #model.save(EMBEDDING_MODEL_FILENAME)

    # Embed edges using Hadamard method
    from node2vec.edges import HadamardEmbedder

    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    # Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
    #edges_kv = edges_embs.as_keyed_vectors()

    # Save embeddings for later use
    #edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)

    return model