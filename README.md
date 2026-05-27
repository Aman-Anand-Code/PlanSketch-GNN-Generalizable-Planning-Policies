# PlanSketch-GNN: Learning Generalizable Planning Policies

PlanSketch-GNN is a simple research-oriented prototype for learning reusable planning policies using Graph Neural Networks (GNNs). The project represents planning states as graphs, trains a GNN to predict the next optimal action, and evaluates whether the learned policy can generalize from small planning instances to larger unseen instances.

This project is inspired by research directions in classical planning, generalized planning, policy learning, graph-based representations, and learning to act and plan.

---

## Repository Name

```text
PlanSketch-GNN-Generalizable-Planning-Policies
```

---

## Short Description

```text
A GNN-based planning prototype that learns reusable action policies from small grid-world planning examples and tests generalization on larger unseen planning instances.
```

---

## Project Motivation

Classical planning focuses on finding a sequence of actions that moves an agent from an initial state to a goal state. In many real-world settings, however, we do not want to solve every new problem completely from scratch. Instead, we want to learn reusable policies that can generalize across different problem instances.

This project explores a simple version of that idea:

- represent a planning state as a graph,
- train a GNN to predict the next useful action,
- test whether the learned policy transfers from small grids to larger unseen grids,
- analyze where learned policies fail during long-horizon rollout.

The project is designed as a lightweight Google Colab prototype.

---

## Research Relevance

This project is related to the following research areas:

- Classical Planning
- Generalized Planning
- Learning General Policies
- Graph Neural Networks for Planning
- Planning State Representation Learning
- Policy Reuse
- Planning and Reinforcement Learning
- Neuro-Symbolic AI
- Machine Learning and Reasoning

The central question explored in this project is:

```text
Can a GNN learn local planning-action patterns from small examples and reuse them on larger unseen planning instances?
```

---

## Dataset / Benchmark Reference

This project uses an automatically generated **Grid Navigation Planning Dataset**.

For standard planning benchmark reference, see:

```text
https://github.com/potassco/pddl-instances
```

The repository above contains PDDL benchmark instances from classical planning competitions. In this project, however, a simplified grid-world planning dataset is generated directly inside the notebook to keep the implementation lightweight and suitable for Google Colab.

---

## Generated Dataset

The notebook automatically generates three datasets:

| Dataset | Grid Sizes | Purpose |
|---|---|---|
| Training Set | 4×4, 5×5, 6×6 | Train the GNN policy |
| Small Test Set | 4×4, 5×5, 6×6 | Test in-distribution performance |
| Large Test Set | 7×7, 8×8, 9×9 | Test generalization to unseen larger grids |

Each sample contains:

- grid layout,
- obstacle positions,
- agent/start position,
- goal position,
- graph representation of the planning state,
- optimal next action computed using BFS.

---

## Planning Domain

The project uses a simple grid navigation domain.

The agent can take four actions:

| Action ID | Action |
|---|---|
| 0 | UP |
| 1 | DOWN |
| 2 | LEFT |
| 3 | RIGHT |

The goal is to move from the start cell to the goal cell while avoiding obstacles.

---

## State Representation

Each grid cell is represented as a graph node.

Edges are created between valid neighboring cells.

Each node has the following features:

| Feature | Meaning |
|---|---|
| `is_obstacle` | Whether the cell is an obstacle |
| `is_agent` | Whether the agent is currently on the cell |
| `is_goal` | Whether the cell is the goal |
| `row_norm` | Normalized row position |
| `col_norm` | Normalized column position |
| `manhattan_to_goal_norm` | Normalized Manhattan distance to the goal |

The planning state is therefore represented as:

```text
Graph = Nodes + Edges + Node Features
```

---

## Model Architecture

The model is a small Graph Convolutional Network.

Architecture:

```text
Input Node Features
        ↓
GCN Layer 1
        ↓
GCN Layer 2
        ↓
GCN Layer 3
        ↓
Agent Node Embedding + Global Graph Embedding
        ↓
Policy Head
        ↓
Action Prediction: UP / DOWN / LEFT / RIGHT
```

The model predicts the next action that should move the agent closer to the goal.

---

## Methodology

The project follows the pipeline below:

```text
Generate Grid Planning Instances
        ↓
Compute Optimal Action using BFS
        ↓
Convert Planning State to Graph
        ↓
Train GNN Policy
        ↓
Evaluate on Small Seen-Size Grids
        ↓
Evaluate on Larger Unseen Grids
        ↓
Run Full Policy Rollout
        ↓
Analyze Generalization and Failure Cases
```

---

## Training Setup

| Component | Value |
|---|---|
| Platform | Google Colab |
| Runtime Used | GPU / CUDA |
| Training Samples | 1200 |
| Small Test Samples | 300 |
| Large Test Samples | 300 |
| Model | 3-layer GCN |
| Hidden Dimension | 64 |
| Optimizer | Adam |
| Loss Function | Cross Entropy Loss |
| Epochs | 20 |

---

## Evaluation Metrics

The project uses the following evaluation metrics:

| Metric | Meaning |
|---|---|
| Training Accuracy | Accuracy on generated training samples |
| Action Prediction Accuracy | Correctness of predicted next action |
| Generalization Gap | Difference between small-grid and large-grid accuracy |
| Rollout Success Rate | Percentage of full rollouts that reach the goal |
| Average Successful Path Length | Average number of steps for successful paths |
| Classification Report | Precision, recall, and F1-score for each action |

---

## Experimental Results

### Training Result

Final training performance:

| Metric | Value |
|---|---:|
| Final Training Accuracy | 70.58% |
| Final Training Loss | 0.7748 |

---

### Action Prediction Accuracy

| Evaluation Setting | Grid Sizes | Accuracy |
|---|---|---:|
| Small seen-size grids | 4×4, 5×5, 6×6 | 71.67% |
| Larger unseen grids | 7×7, 8×8, 9×9 | 66.33% |
| Generalization Gap | Small → Large | 5.33% |

---

## Result Interpretation

The model achieved **71.67% action prediction accuracy** on small seen-size grids and **66.33% accuracy** on larger unseen grids.

This shows that the GNN policy learned useful local planning patterns that transfer moderately well to larger grid instances. The generalization gap of **5.33%** suggests that the learned representation does not completely collapse when grid size increases.

However, full rollout success was lower. This indicates that one-step action prediction alone is not sufficient for reliable long-horizon planning because small action errors can accumulate and lead to loops.

---

## Rollout Evaluation

| Grid Size | Success Rate | Average Successful Path Length |
|---|---:|---:|
| 7×7 | 20.00% | 3.60 |
| 8×8 | 16.00% | 3.12 |

---

## Failure Analysis

During rollout, the learned policy sometimes entered loops.

Example failed path:

```text
Start: (0, 3)
Goal:  (0, 6)

Predicted path:
(0, 3) → (1, 3) → (2, 3) → (1, 3) → (2, 3) → ...
```

This shows a common weakness of pure one-step learned policies: even if action prediction accuracy is reasonable, small mistakes can cause repeated states and prevent goal completion.

---

## Key Observation

The project demonstrates a useful difference between:

```text
Action Prediction Accuracy
```

and

```text
Long-Horizon Planning Success
```

A model can perform reasonably well on one-step action prediction but still fail in full planning rollout. This motivates future work on combining learned policies with symbolic planning, replanning, or sketch-guided decomposition.

---

## How to Run

### Recommended: Google Colab

Open Google Colab and select:

```text
Runtime → Change runtime type → T4 GPU
```

Then run the notebook.

The code can also run on CPU, but GPU is recommended for faster training.

---

## Installation

Install the dependencies:

```bash
pip install torch networkx pandas matplotlib scikit-learn tqdm
```

---

## Suggested `requirements.txt`

```text
torch
networkx
pandas
matplotlib
scikit-learn
tqdm
numpy
```

---

## Project Structure

```text
PlanSketch-GNN-Generalizable-Planning-Policies/
│
├── README.md
├── PlanSketch_GNN_Colab.ipynb
├── requirements.txt
│
├── results/
│   ├── plansketch_gnn_training_history.csv
│   ├── plansketch_gnn_evaluation_summary.csv
│   └── plansketch_gnn_rollout_results.csv
│
└── assets/
    └── rollout_visualization.png
```

---

## Output Files

The notebook saves three CSV files:

### 1. Training History

```text
plansketch_gnn_training_history.csv
```

Contains:

- epoch,
- training loss,
- training accuracy.

### 2. Evaluation Summary

```text
plansketch_gnn_evaluation_summary.csv
```

Contains:

- small-grid test accuracy,
- large-grid test accuracy,
- generalization gap.

### 3. Rollout Results

```text
plansketch_gnn_rollout_results.csv
```

Contains:

- rollout success rate,
- average successful path length.

---

## Key Features

- Generates a planning dataset automatically
- Represents planning states as graphs
- Uses BFS to compute optimal one-step actions
- Trains a GNN policy for action prediction
- Tests generalization from small to larger grids
- Evaluates action accuracy and rollout success
- Visualizes learned policy rollout
- Saves training and evaluation results as CSV files
- Runs on free Google Colab GPU

---

## Limitations

This is a first prototype and has the following limitations:

1. The domain is limited to grid navigation.
2. The model learns one-step action prediction, not full planning.
3. Rollout success is still low on larger grids.
4. The learned policy may enter loops.
5. The dataset is generated synthetically.
6. The model does not yet use symbolic planning constraints.
7. The system does not yet implement sketch-guided decomposition.
8. The current version does not parse full PDDL benchmark files.

---

## Future Improvements

Possible future extensions:

- Add loop detection and replanning.
- Combine the GNN policy with a symbolic planner.
- Add sketch-guided planning decomposition.
- Extend the dataset to Blocksworld and Gripper.
- Parse standard PDDL planning benchmarks.
- Compare GNN policies with BFS, A*, and heuristic planners.
- Add Graph Attention Networks.
- Add curriculum learning from small to large planning instances.
- Add reinforcement learning fine-tuning.
- Improve rollout success with memory or visited-state features.
- Add interpretability analysis for important nodes and graph regions.

---

## Resume Bullet

```text
Implemented a GNN-based planning prototype that represents grid-world planning states as graphs and learns reusable action policies from small examples, achieving 71.67% action accuracy on seen-size grids and 66.33% on larger unseen grids.
```

---

## Suggested GitHub Topics

```text
classical-planning
generalized-planning
graph-neural-networks
gnn
planning
reinforcement-learning
machine-learning
reasoning
pytorch
networkx
gridworld
policy-learning
ai-planning
neuro-symbolic-ai
```

---

## Author

**Aman Anand**

B.Tech Computer Science, KIIT University  
B.S. Data Science and Applications, IIT Madras  

GitHub:

```text
https://github.com/Aman-Anand-Code
```

---

## License

This project can be released under the MIT License.

Recommended license:

```text
MIT License
```

---

## Acknowledgements

This project is an educational and research-prototype implementation inspired by work in classical planning, generalized planning, graph-based policy learning, and machine learning for reasoning.
