- base: 
  - model: GAT
  - sampling: random edge sampling 
  - acc: 0.8286

- [x] mid try: random node Sampling
  - https://distill.pub/2021/gnn-intro/ 

- X strong try: whole graph
  - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn2_cora.py

- [x] 1- Loss func:
  - a * crossEntropy + (1 - a) * cosineSimilaryScore 
- [x] 2- add residule
  - not improve

- find out
  - why it not work (GAT overfitting)

- [x] soft vote (final 0.83370)
  - GCNConv
    - Semi-Supervised Classification with Graph Convolutional Networks
    - https://arxiv.org/abs/1609.02907
  -  ~~X GATv2Conv ~~
  - ~~X GATConv~~
    - Graph Attention Networks
    - https://arxiv.org/abs/1710.10903
  - ~~X AGNNConv~~
    - Attention-based Graph Neural Network for Semi-supervised Learning
    - https://arxiv.org/abs/1803.03735
  - ~~X SAGEConv~~
    - Inductive Representation Learning on Large Graphs
    - https://arxiv.org/abs/1706.02216
  - ~~use GCNII (adj graph so big)~~
    - Paper: [GCNII](https://txyz.ai/paper/81a6391b-8171-41be-a2b6-4b1e748c18b1)
      - Expressive Power
      - Initial Residual Connection
      - Identity Mapping
    