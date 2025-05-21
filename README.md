# Fourier_based-MPNN
![Fourier-based_Message_Passing](MSH_GNN (1).jpg)
## Envioronment
- Python >= 3.8  
- PyTorch >= 1.12  
- PyTorch Geometric  
- RDKit (if applicable)  
- CUDA >= 11.0 (for GPU training)  


## Dataset
###  Homophilic Graph Datasets
"AmazonComputers", "AmazonPhoto", "CoauthorCS", "CoauthorPhysics" datasets:

https://github.com/shchur/gnn-benchmark/raw/master/data/npz/

###  Heterophilic Graph Datasets
"amazon-ratings", "minesweeper", "tolokers", "questions" datasets :

https://github.com/yandex-research/heterophilous-graphs/tree/main/data

```bash
python download_dataset.py 
```
## Submit Script
### For single GPU:

```bash
python main.py --dataset NCI1
```
### For multi-GPU:
```bash
torchrun --nproc_per_node=2 multi_gpu_main.py --dataset COLLAB
```

## Results
### Table: Graph classification results on small-scale benchmarks (Accuracy %)

| Method                          | PROTEINS       | NCI1           | IMDB-B         | IMDB-M         | COLLAB         |
|---------------------------------|----------------|----------------|----------------|----------------|----------------|
| #Graphs                         | 1,113          | 4,110          | 1,000          | 1,500          | 5,000          |
| #Nodes                          | 39.06          | 29.87          | 19.77          | 13.00          | 74.49          |
| #Edges                          | 145.60         | 64.60          | 193.10         | 131.87         | 4,914.4        |
| #Node Features                  | 3              | 0              | 0              | 0              | 0              |
| PATCHY-SAN                      | 75.00 ±2.51    | 78.60 ±1.91    | 71.00 ±2.29    | 45.23 ±2.84    | 72.60 ±2.15    |
| GCN                             | 73.24 ±0.73    | 76.29 ±1.11    | 73.26 ±0.90    | 50.39 ±0.41    | 80.59 ±0.87    |
| PG                              | 76.80 ±1.80    | 82.00 ±0.30    | 76.80 ±2.60    | 53.20 ±0.30    | 80.00 ±0.80    |
| CoCN                            | 76.86 ±0.13    | 82.89 ±1.10    | 77.26 ±0.60    | 56.32 ±0.18    | 86.15 ±0.10    |
| GIN                             | 73.84 ±4.46    | 76.62 ±1.41    | 72.78 ±0.96    | 48.13 ±1.36    | 78.19 ±0.63    |
| └─ +pseudo node                | 74.11 ±1.42    | 77.08 ±1.49    | --             | --             | --             |
| GraphSAGE                       | 73.48 ±1.56    | 83.22 ±1.87    | 68.80 ±1.54    | 47.60 ±3.50    | 73.90 ±1.70    |
| └─ +pseudo node                | 73.93 ±1.85    | 74.31 ±2.17    | --             | --             | --             |
| DiffPool                        | 75.62 ±5.18    | 76.62 ±1.93    | 73.14 ±0.70    | 51.31 ±0.72    | 82.13 ±0.43    |
| └─ +pseudo node                | 75.98 ±1.89    | 77.08 ±1.31    | --             | --             | --             |
| ToFKPool                        | 70.48 ±1.01    | 67.02 ±2.25    | 71.58 ±0.95    | 48.59 ±0.72    | 77.58 ±0.84    |
| SAGPool                         | 71.15 ±1.64    | 76.52 ±2.55    | 72.55 ±0.82    | 50.20 ±0.44    | 78.03 ±0.31    |
| StructPool                      | 75.16 ±0.36    | 78.64 ±1.53    | 72.06 ±1.60    | 50.23 ±0.63    | 77.27 ±0.70    |
| SEP                             | 76.42 ±0.91    | 82.76 ±0.61    | 72.16 ±1.52    | 51.53 ±0.86    | 81.28 ±0.15    |
| GMT                             | 75.09 ±0.59    | 76.35 ±2.22    | 73.48 ±0.66    | 50.66 ±0.82    | 80.74 ±0.54    |
| N² (Sun et al. 2024)           | 77.53 ±1.78    | 83.52 ±1.75    | 79.95 ±2.46    | 57.31 ±2.19    | 86.72 ±1.62    |
| **FourierMP (Ours)**           | **78.69±2.01** | **88.59±3.34** | **86.28±0.74** | **61.07±0.38** |                |


## Reference code
https://github.com/sunjss/N2

