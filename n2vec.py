import networkx as nx
from node2vec import Node2Vec
from numpy.random import seed

# Accepts a networkx graph and returns a its embedding using Node2Vec
def embed_graphs(graph, embedding_filename="", embedding_model_filename="", edges_embedding_filename="", embedding_dir="graph_embed_data/", save_embed_files_b=True):
    EMBEDDING_FILENAME = embedding_dir + embedding_filename + ".emb" 
    EMBEDDING_MODEL_FILENAME = embedding_dir + embedding_model_filename + ".model" 
    EDGES_EMBEDDING_FILENAME = embedding_dir + edges_embedding_filename + ".emb" 

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(
        graph, dimensions=32, q=0.1, walk_length=30, num_walks=80, quiet=True, workers=4#, seed=seed(1)
    ) 

    # Embed nodes
    model = node2vec.fit(
        window=10, min_count=1, batch_words=4
    )  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Embed edges using Hadamard method
    from node2vec.edges import HadamardEmbedder

    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    if save_embed_files_b:
        save_embed_files(EMBEDDING_FILENAME, EDGES_EMBEDDING_FILENAME, EMBEDDING_MODEL_FILENAME, model, edges_embs)

    return model


def save_embed_files(embed_filename, edge_embed_filename, model_embed_filename, nmodel, emodel):
    # Save embeddings for later use
    nmodel.wv.save_word2vec_format(embed_filename)

    # Save model for later use
    #nmodel.save(model_embed_filename)

    # Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
    #edges_kv = emodel.as_keyed_vectors()

    # Save embeddings for later use
    #edges_kv.save_word2vec_format(edge_embed_filename)