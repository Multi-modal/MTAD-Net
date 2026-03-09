# MTAD-Net: Multimodal Traffic Anomaly Detection Network

## MTAD-Net

This repository contains the code for the paper, Multimodal Learning for Anomaly Detection in Network Traffic
(The code is being sorted out and we will continue to update it.)

##  Overview

MTAD-Net integrates three complementary data sources to achieve a comprehensive characterization of network behavior:
1. Temporal Traffic Patterns: A temporal encoder uses patch-based processing and a memory mechanism to capture local and global variations.
2. Routing Connectivity (Topology): A topology encoder employs a Graph Attention Network (GAT) to model inter-router dependencies and generate global graph embeddings.
3. Network Environment Information: A text encoder uses a frozen Large Language Model (LLM) to extract semantic context from structured prompts detailing node information, routing policies, and traffic statistics.

These modalities are fused using a cross-modal attention mechanism and adaptive gated weighting, enabling robust anomaly identification through reconstruction error calculation.
## Datasets
The MTAD-Net framework was evaluated on two real-world backbone network datasets:

1. Abilene Dataset: Contains traffic measurement data collected from the Internet2 backbone network in the United States. It monitors 12 core routers and 144 origin-destination (OD) flows, capturing patterns like periodic fluctuations, congestion events, and anomalies.
2. GÉANT Dataset: Contains traffic matrices from the GÉANT backbone network, connecting research institutions across Europe. It includes 23 routers and 529 OD flows, featuring complex spatial dependencies.

## How to run

- Train and detect:

> python anomaly_detection-train.py --dataset Abilene
>
> Then you will train the whole model and will get the detected score.

## How to run with your own data

- By default, datasets are placed under the "dataset" folder. If you need to change the dataset, you can modify the dataset path in the main file.

> python main.py  --'dataset'  your dataset


