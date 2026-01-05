# Research Strategy

ManualsGraph is built on the premise that troubleshooting information is inherently hierarchical and causal.

## Adapting "GraphReasoning"

Our approach is inspired by Markus J. Buehler's work on scientific discovery through knowledge extraction. However, we adapt this framework for industrial maintenance:

| Aspect | Scientific Papers (Buehler) | Industrial Manuals (Our Case) |
| :--- | :--- | :--- |
| **Topology** | Scale-free, highly connected | Directed Acyclic Graph (DAG) |
| **Logic** | Exploratory / Hypothesis | Deterministic / Troubleshooting |
| **Content** | Abstract concepts | Symptoms, Causes, Fixes |
| **Format** | Dense Text | Multimodal (Tables, Diagrams, Text) |

## Knowledge Graph Schema

We extract nodes based on a specific schema:

-   **SYMPTOM**: Observable manifestations (error codes, sensor patterns).
-   **ROOT_CAUSE**: The underlying failure mode.
-   **FIX**: The repair procedure or corrective action.
-   **COMPONENT**: The physical part involved.

## Why DAG over Scale-Free?

Industrial troubleshooting follows a **top-down deterministic logic**. A scale-free graph (with many cross-domain connections) would introduce cycles and "creative" reasoning that could lead to incorrect or unsafe diagnostic advice. By enforcing a DAG structure, we ensure that:
1.  Symptoms clearly lead to Causes.
2.  Causes lead to Fixes.
3.  The reasoning is traceable and explainable.

## Further Reading
For a deep dive into the extraction methodology, see the original `ANALYSIS-KG-Extraction-Strategy.md` file in the root directory.
