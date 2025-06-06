import numpy as np
import openai

# ──────────────────────────────────────────────────────────────────────────────
# (A) Suppose you have these from your previous run:
#
#   embeddings : np.ndarray shape (N, d)
#       – each row i is the normalized embedding of DataFrame row idx_list[i]
#   idx_list   : list of length N
#       – idx_list[i] is the DataFrame index/label corresponding to embeddings[i]
#   G          : networkx.Graph
#       – nodes are exactly the values in idx_list (strings like "WO0001", etc.)
#
#   And you have your OpenAI API key set:
#     openai.api_key = os.getenv("OPENAI_API_KEY")
# ──────────────────────────────────────────────────────────────────────────────


def find_best_node_and_neighbors(new_problem: str,
                                 embeddings: np.ndarray,
                                 idx_list: list,
                                 G: 'networkx.Graph',
                                 model_name: str = "text-embedding-ada-002"
                                ):
    """
    Given a new free-text `new_problem`, embed it, find the single closest
    existing node by cosine similarity against `embeddings`, and then
    return that node plus all of its direct neighbors in G.
    
    Inputs:
      new_problem : str
        A free‐text problem description (you can also append “| solution: ” +
        an empty string or some existing solution, but here we assume you only
        have a new problem). We’ll embed it as “problem: {new_problem} | solution: ”.
      
      embeddings  : np.ndarray, shape (N, d)
        The normalized embeddings of the N rows (in the same order as idx_list).
      
      idx_list    : list of length N
        idx_list[i] is the DataFrame index (string or int) corresponding to embeddings[i].
      
      G           : networkx.Graph
        The graph over those same N nodes.  Every node in G is one of idx_list.
      
      model_name  : str (optional)
        Which OpenAI embedding model to use (must match what you used originally).
    
    Returns:
      best_idx    : int
        The integer position (0 ≤ best_idx < N) of the closest existing node.
      
      best_label  : same type as idx_list[0]
        The actual DataFrame index/label of the closest node (i.e. idx_list[best_idx]).
      
      neighbors   : list
        A list of all direct neighbors of best_label in G (graph‐neighbors), in arbitrary order.
    """
    # 1) Build the same “prompt” format you used when embedding each row:
    prompt = f"problem: {new_problem} | solution: "
    
    # 2) Call the OpenAI embeddings endpoint (new‐style API) to get a vector of dimension d
    response = openai.embeddings.create(
        model=model_name,
        input=[prompt]
    )
    # The new response object is not subscriptable; use response.data
    new_vec = np.array(response.data[0].embedding, dtype=float)
    new_vec /= (np.linalg.norm(new_vec) + 1e-12)
    
    # 3) Compute cosine similarities against all existing embeddings:
    #    Since `embeddings` rows are already unit‐normalized, cosine = dot product
    sims = embeddings.dot(new_vec)   # shape (N,)
    
    # 4) Find the index of the maximum similarity
    best_idx = int(np.argmax(sims))
    best_label = idx_list[best_idx]
    
    # 5) Gather all direct neighbors of that best_label in G
    #    (May be empty if it had no edges; in practice InfoLLMGraph always ensures at least one edge.)
    neighbors = list(G.neighbors(best_label))
    
    return best_idx, best_label, neighbors
