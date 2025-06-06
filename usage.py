# ──────────────────────────────────────────────────────────────────────────────
# Example Usage
# ──────────────────────────────────────────────────────────────────────────────
import os
import math
import pandas as pd
import numpy as np
import networkx as nx
import openai
import random

from InfoLLMGraph import InfoLLMGraph
from best_neighbors import find_best_node_and_neighbors


# Ensure your OpenAI API key is set in the environment:
#   export OPENAI_API_KEY="your_token_here"
openai.api_key = ''

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of work orders
n_work_orders = 200

# Generate work order numbers: WO0001, WO0002, ..., WO0200
work_order_numbers = [f"WO{str(i).zfill(4)}" for i in range(1, n_work_orders + 1)]

# Generate sensor column names
pressure_probes = [f"PP_{str(i).zfill(2)}" for i in range(1, 6)]      # PP_01 .. PP_05
temp_sensors   = [f"TS_{str(i).zfill(2)}" for i in range(1, 11)]     # TS_01 .. TS_10
valves         = [f"V_{str(i).zfill(2)}"  for i in range(1, 50)]     # V_01 .. V_49

# Prepare lists of creative problem/solution pairs (with alarm codes)
problem_solution_pairs = [
    {
        "problem": "DP-ALARM-102: Differential pressure across Filter A (PP_03 vs PP_04) exceeded threshold",
        "solution": "Replace Filter A element, bleed down pressure, and reset DP-ALARM-102"
    },
    {
        "problem": "TS-ALARM-07: Overtemperature at TS_08 (reading > 85°C) indicating potential bypass valve V_12 stuck",
        "solution": "Inspect V_12, verify actuator function, clear obstruction, and restart circulation pump"
    },
    {
        "problem": "V-FAULT-015: Valve V_25 failed to respond to open command (remote actuator error)",
        "solution": "Check actuator power supply, replace V_25 solenoid coil, and cycle the valve"
    },
    {
        "problem": "PP-LOW-ALM-05: Low upstream pressure at PP_02 (reading < 5 psi), possibly clogged inlet strainer",
        "solution": "Shut down skid, clean inlet strainer, verify PP_02 calibration, and restart skid"
    },
    {
        "problem": "TS-UNDERHEAT-03: TS_05 reading below 10°C, heat exchanger HX-1 control loop malfunction",
        "solution": "Check HX-1 control valve (V_33), recalibrate TS_05, and adjust heater control PID parameters"
    },
    {
        "problem": "V-LEAK-ALM-022: Detected leakage at valve V_41 (seal failure, TS_10 elevated)",
        "solution": "Isolate V_41, replace gasket, torque body bolts to spec, and verify no further leakage"
    },
    {
        "problem": "PP-SPIKE-ALM-009: Pressure spike recorded between PP_05 and PP_01 indicating rapid valve V_07 closure",
        "solution": "Review control sequence for V_07, install soft-start CV, and reset PP-SPIKE-ALM-009"
    },
    {
        "problem": "TS-DRIFT-ALM-011: TS_02 drifted > ±3°C from baseline over 24h, sensor calibration suspected",
        "solution": "Calibrate TS_02 against reference, replace if calibration out of tolerance (<±1°C), clear alarm"
    },
    {
        "problem": "V-STUCK-ALM-018: Valve V_18 positioner not reaching commanded position, causing flow imbalance",
        "solution": "Lubricate valve stem, inspect positioner signals, recalibrate positioner, and cycle function"
    },
    {
        "problem": "TS-OVERRANGE-ALM-019: TS_09 reading saturated at 120°C, possible thermocouple fault",
        "solution": "Replace TS_09 thermocouple, verify junction box wiring, and test at known temperature block"
    },
]

# Function to randomly pick a problem/solution
def pick_problem_solution():
    pair = random.choice(problem_solution_pairs)
    return pair["problem"], pair["solution"]

# Create empty dictionary to hold columns
data = {}

# Populate problem and solution columns
problems = []
solutions = []
for _ in range(n_work_orders):
    prob, sol = pick_problem_solution()
    problems.append(prob)
    solutions.append(sol)

data["problem"]  = problems
data["solution"] = solutions

# Create DataFrame
df = pd.DataFrame(data, index=work_order_numbers)

# Set index name
df.index.name = "work_order_no"

# Build the graph with alpha=0.5, threshold tau=0.75
G, S_info, S_emb, feature_weights,embeddings, idx_list = InfoLLMGraph(df, alpha=0.5, tau=0.75)

# Inspect the resulting graph
print("Nodes:", G.nodes())
print("Edges with weights:")
for u, v, attrs in G.edges(data=True):
    print(f"  {u} -- {v}, weight={attrs['weight']:.4f}")

# If you want to retrieve S_info or S_emb for further analysis:
print("Feature weights (w):", feature_weights)
print("S_info matrix:\n", S_info)
print("S_emb matrix:\n", S_emb)

    # 2) Define your new problem text
new_problem_text = "TS-ALARM-07 triggered: Overtemperature at TS_08 > 85°C, suspect valve blockage"

# 3) Call the function:
best_idx, best_label, neighbors = find_best_node_and_neighbors(
    new_problem=new_problem_text,
    embeddings=embeddings,
    idx_list=idx_list,
    G=G,
    model_name="text-embedding-ada-002"
)

print(f"Closest existing node index = {best_idx}, label = {best_label}")
row = df.loc[neighbors]

#Then print both fields however you like:
print(f"Closest nodes text: problem={row['problem']!r}, solution={row['solution']!r}")
print("Its graph‐neighbors are:", neighbors)
