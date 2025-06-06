# InfoLLMGraph

## Motivation

When troubleshooting technical systems—whether industrial filtration skids, manufacturing lines, or complex machinery—engineers often rely on historical work orders, incident reports, and troubleshooting logs. A pure embedding-based search (e.g. computing cosine similarity on “problem + solution” text) can miss records that share a critical rare flag (for example, a specific alarm code or component ID) but describe it in different words. Conversely, exact keyword matching on alarm codes alone loses the nuance of natural-language variation.

**InfoLLMGraph** is designed to bridge this gap by combining:

1. **Information-theoretic overlap** on discrete matches (e.g. identical alarm codes, valve IDs, or other rare tokens), and  
2. **LLM-based semantic embeddings** of the full “problem + solution” text.  

The result is a single, weighted graph of all work‐order records. Each node represents one row; edges carry a score between 0 and 1 that reflects a mix of:
- **Exact match on rare feature** (information overlap), and  
- **Soft semantic similarity** (embedding cosine).  

By retrieving “closest node + its neighbors” from this graph, you surface both semantically relevant records and any that share a rare, business‐critical flag—even if the wording differs. This increases the chance of finding truly relevant technical records that embeddings alone might miss.

---

## Approach Overview

1. **Input DataFrame**  
   - Must contain exactly two columns:  
     - `problem` — a free-text description of the issue.  
     - `solution` — a free-text description of how it was resolved.  
   - The DataFrame’s index (e.g. `"WO0001"`, `"WO0123"`) becomes the node label for each record.

2. **Information‐Overlap Component**  
   - Treat `problem` and `solution` as categorical features (each unique string is a “value”).  
   - Compute empirical entropy for each column:  
     ```
     H_f = –∑ [ P(value) * log(P(value)) ]  
     ```
     where P(value) = (frequency of that exact string) / N.  
   - Define a raw feature weight:  
     ```
     w_f_raw = 1 / (H_f + ε)
     ```  
     where ε > 0 is a small constant (e.g. 1e-12).  
   - Normalize so that the two weights sum to 1:  
     ```
     w_problem + w_solution = 1
     ```  
   - For any two rows i, j, compute:  
     ```
     S_info(i, j) = 
       w_problem * 1[problem_i == problem_j]
       + w_solution * 1[solution_i == solution_j]
     ```  
     If two rows share the exact same “problem” text, they get w_problem; if they share the exact same “solution” text, they get w_solution. In typical datasets where every row’s text is unique, off-diagonal S_info(i, j) = 0.

3. **Embedding Component**  
   - For each row, build a single prompt string:  
     ```
     "problem: {row.problem} | solution: {row.solution}"
     ```  
   - Call OpenAI’s `text-embedding-ada-002` endpoint to get a 1536-dimensional vector, then L2-normalize it to unit length.  
   - For any two normalized vectors e_i, e_j, define:  
     ```
     S_emb(i, j) = dot(e_i, e_j)   # cosine similarity in [0, 1]
     ```

4. **Combine Scores**  
   - Choose mixing parameter α ∈ [0, 1] (default α = 0.5).  
   - Define the final similarity matrix:  
     ```
     S(i, j) = α * S_info(i, j) + (1 – α) * S_emb(i, j)
     ```  
     By construction, 0 ≤ S(i, j) ≤ 1.

5. **Graph Construction (Threshold τ)**  
   - Initialize an undirected NetworkX graph `G` with all N nodes labeled by the DataFrame’s index.  
   - For every pair (i, j), if S(i, j) ≥ τ (threshold, e.g. 0.75), add an edge `(node_i, node_j, weight=S(i, j))`.  
   - If any node is still isolated (degree 0), forcibly connect it to its single highest‐score neighbor to guarantee full connectivity.

6. **Output**  
   - `G` — a weighted, undirected graph where each node corresponds to one DataFrame index, and each edge has attribute `weight` ∈ (0, 1].  
   - `S_info` — an N×N NumPy array of information‐overlap scores.  
   - `S_emb` — an N×N NumPy array of semantic cosine similarities.  
   - `w` — a dictionary mapping `{"problem": weight, "solution": weight}` to the normalized entropy weights.  
   - Optionally, also return the raw `embeddings` array (shape N×1536) and `idx_list` (list of node labels in the same order) for retrieval later.

---

## Why Combine Info Theory + Embeddings?

- **Embedding-Only Retrieval Misses Rare Flags**  
  Two work orders may both log “DP-ALARM-102,” but one’s text might read “High differential pressure across filter A” while another says “Filter A ΔP exceeded safe limit.” An embedding model alone could assign a lower cosine to that exact alarm code match versus other more generic records.  
- **Info-Only Misses Paraphrase Nuance**  
  If every row’s “problem” and “solution” text is guaranteed unique, S_info collapses to purely diagonal (only self-matches). You’ll lose the ability to catch semantically similar records.  
- **Tunable Balance (α)**  
  If your domain places heavy importance on exact alarm codes or rare component IDs, set α near 1. If you care more about free-text semantics, set α near 0. The default α = 0.5 balances both.  
- **Interpretability & Rarity Emphasis**  
  If two rows share a rare value (e.g. an alarm code that appears only twice), the information‐overlap component gives disproportionate weight—this mimics the idea of pointwise mutual information.  
- **Connected Graph for Richer Retrieval**  
  By requiring connectivity (even if forced for isolated nodes), you can later run clustering, Personalized PageRank, or simply walk neighbors-of-neighbors to find a small subgraph that contextualizes a query.

---

## Installation

1. Clone this repository or copy these two files into your project:
   - `InfoLLMGraph.py`
   - `best_neighbors.py`

2. Install dependencies:
   ```bash
   pip install pandas numpy networkx openai

## Usage

Check usage.py for an example.
