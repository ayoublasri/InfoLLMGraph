import os
import math
import pandas as pd
import numpy as np
import networkx as nx
import openai
import random

# ──────────────────────────────────────────────────────────────────────────────
# InfoLLMGraph: Build a weighted graph over DataFrame rows using
#               information‐overlap on 'problem'/'solution' plus LLM embeddings.
# 
# Requirements:
#   pip install pandas numpy networkx openai
#
# Inputs:
#   df: pandas DataFrame with exactly two columns, 'problem' and 'solution'.
#       The index of df will become the graph’s node labels.
#   alpha: mixing parameter in [0,1] (default 0.5).
#   tau:  threshold in [0,1] for edge inclusion (default 0.75).
#
# Output:
#   G:    networkx.Graph where each node is one df.index, and each edge has attribute 'weight' in (0,1].
#   S_info:  numpy array (N×N) of information‐overlap scores.
#   S_emb:   numpy array (N×N) of embedding‐cosine similarities.
#   w:       dict mapping column name → normalized feature weight.
#
# Usage Example:
#   openai.api_key = os.getenv("OPENAI_API_KEY")
#   G, S_info, S_emb, w = InfoLLMGraph(df, alpha=0.5, tau=0.75)
# ──────────────────────────────────────────────────────────────────────────────

def InfoLLMGraph(df: pd.DataFrame, *, alpha: float = 0.5, tau: float = 0.75):
    """
    Build InfoLLMGraph from a DataFrame with columns 'problem' and 'solution'.
    Uses OpenAI embeddings to compute S_emb, and information‐overlap on exact
    matches of 'problem'/'solution' to compute S_info. Returns a connected graph G.
    """
    # Verify input
    required_cols = ["problem", "solution"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain exactly columns: {required_cols}")
    N = len(df)
    if N == 0:
        raise ValueError("DataFrame is empty.")
    
    # Map df.index ↔ integer node IDs 0..N-1
    idx_list = list(df.index)
    id_from_idx = {idx: i for i, idx in enumerate(idx_list)}
    idx_from_id = {i: idx for i, idx in enumerate(idx_list)}
    
    # ──────────────────────────────────────────────────────────────────────────
    # 1) COMPUTE PER‐FEATURE ENTROPY AND NORMALIZED WEIGHTS
    #    Here features are the two columns: 'problem' and 'solution', treated as categorical.
    # ──────────────────────────────────────────────────────────────────────────
    
    epsilon = 1e-12
    w_raw = {}
    H = {}
    # For each column, treat its distinct text values as categories
    for col in required_cols:
        # Compute empirical probabilities of each distinct value
        counts = df[col].value_counts(normalize=True)
        probs = counts.values  # array of relative frequencies
        # Entropy H_f = -∑ p log p  (use log base 2 or natural; choice doesn't affect ranking)
        H_f = -np.sum(probs * np.log(probs + 1e-16))
        H[col] = H_f
        w_raw[col] = 1.0 / (H_f + epsilon)
    # Normalize weights so they sum to 1
    total_w = sum(w_raw.values())
    w = {col: w_raw[col] / total_w for col in required_cols}
    
    # ──────────────────────────────────────────────────────────────────────────
    # 2) BUILD S_info MATRIX (N×N) VIA INVERTED‐INDEX ON EXACT MATCHES
    # ──────────────────────────────────────────────────────────────────────────
    
    S_info = np.zeros((N, N), dtype=float)
    # For each feature column, build a mapping: value → list of row‐IDs that have that value
    for col in required_cols:
        # Create a dict: value → [list of integer IDs]
        val_to_ids = {}
        for idx, val in enumerate(df[col].tolist()):
            val_to_ids.setdefault(val, []).append(idx)
        # For each group of rows sharing that exact value, add w[col] to all pairwise S_info[i,j]
        for ids in val_to_ids.values():
            if len(ids) < 2:
                continue
            weight_f = w[col]
            # Add weight_f to each unordered pair (i, j) with j>i
            for i in ids:
                for j in ids:
                    if j > i:
                        S_info[i, j] += weight_f
    # Mirror the upper‐triangle into the lower‐triangle to make S_info symmetric
    iu, ju = np.triu_indices(N, k=1)
    S_info[ju, iu] = S_info[iu, ju]
    # Each row matches itself on both columns, so set diagonal = 1.0
    np.fill_diagonal(S_info, 1.0)
    
    # ──────────────────────────────────────────────────────────────────────────
    # 3) COMPUTE LLM EMBEDDINGS FOR EACH ROW → S_emb
    # ──────────────────────────────────────────────────────────────────────────
    
    # Build a list of concatenated prompts: "problem: <...>  solution: <...>"
    prompts = []
    for idx in idx_list:
        p = df.at[idx, "problem"] or ""
        s = df.at[idx, "solution"] or ""
        prompt = f"problem: {p} | solution: {s}"
        prompts.append(prompt)
    
    # Call OpenAI Embedding API in batches (batch‐size ≤1000 to avoid rate limits)
    # Model: "text-embedding-ada-002"
    model_name = "text-embedding-ada-002"
    batch_size = 100
    # ada‐002 embeddings have dimension 1536
    embeddings = np.zeros((N, 1536), dtype=float)
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = prompts[start:end]
        response = openai.embeddings.create(
            model=model_name,
            input=batch
        )
        # Extract embeddings and normalize
        for i, obj in enumerate(response.data):
            vec = np.array(obj.embedding, dtype=float)
            norm = np.linalg.norm(vec) + 1e-12
            embeddings[start + i] = vec / norm
    
    # Now embeddings is shape (N, d). Compute S_emb = embeddings ⋅ embeddings^T  (cosine similarity)
    S_emb = embeddings.dot(embeddings.T)
    # Due to numerical rounding, ensure any tiny negatives are clipped to 0
    np.clip(S_emb, 0.0, 1.0, out=S_emb)
    
    # ──────────────────────────────────────────────────────────────────────────
    # 4) COMBINE INTO FINAL SCORE MATRIX S = α ⋅ S_info + (1−α) ⋅ S_emb
    # ──────────────────────────────────────────────────────────────────────────
    
    S = alpha * S_info + (1.0 - alpha) * S_emb
    # Guarantee values in [0,1]
    np.clip(S, 0.0, 1.0, out=S)
    
    # ──────────────────────────────────────────────────────────────────────────
    # 5) BUILD THE CONNECTED GRAPH G USING THRESHOLD τ
    # ──────────────────────────────────────────────────────────────────────────
    
    G = nx.Graph()
    # Add all nodes labeled by the original df.index
    G.add_nodes_from(idx_list)
    
    # Add edges for all pairs with S[i,j] ≥ tau
    for i in range(N):
        for j in range(i + 1, N):
            if S[i, j] >= tau:
                u = idx_from_id[i]
                v = idx_from_id[j]
                G.add_edge(u, v, weight=float(S[i, j]))
    
    # If any node is isolated (degree 0), connect it to its highest‐score neighbor
    for i in range(N):
        node_i = idx_from_id[i]
        if G.degree(node_i) == 0:
            # Find j_best = argmax_{j ≠ i} S[i, j]
            sims = S[i].copy()
            sims[i] = -1.0
            j_best = int(np.argmax(sims))
            node_j = idx_from_id[j_best]
            G.add_edge(node_i, node_j, weight=float(S[i, j_best]))
    
    return G, S_info, S_emb, w, embeddings, idx_list
