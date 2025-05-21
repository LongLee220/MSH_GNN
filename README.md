# Architecture
![message_passing](message_passing.jpg)
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

| Methods        | MUTAG       | PTC         | PROTEINS    | NCI1        | IMDB-B      | IMDB-M      | RDT-B       | COLLAB      |
|----------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| GNTK           | 90.0±8.5    | 67.9±6.9    | 75.6±4.2    | 84.2±1.5    | 76.9±3.6    | 52.8±4.6    | N/A         | 83.6±1.0    |
| DCNN           | N/A         | N/A         | 61.3±1.6    | 56.6±1.0    | 49.1±1.4    | 33.5±1.4    | N/A         | 52.1±0.7    |
| DGCNN          | 85.8±1.8    | 58.6±2.2    | 75.5±0.9    | 74.4±0.5    | 70.0±0.9    | 47.8±0.9    | N/A         | 73.8±0.5    |
| IGN            | 83.9±13.0   | 58.5±6.6    | 76.6±5.5    | 74.3±2.7    | 72.0±5.5    | 48.7±3.4    | N/A         | 78.3±2.5    |
| GIN            | 89.4±5.6    | 64.6±1.0    | 76.2±2.8    | 82.7±1.7    | 75.1±5.1    | 52.3±2.8    | 92.4±2.5    | 80.2±1.9    |
| PPGNs          | 90.6±8.7    | 66.2±6.6    | 77.2±4.7    | 83.2±1.1    | 73.0±5.8    | 50.5±3.6    | N/A         | 81.4±1.4    |
| Natural GN     | 89.4±1.6    | 66.8±1.1    | 71.7±1.0    | 82.4±1.3    | 73.5±2.0    | 51.3±1.5    | N/A         | N/A         |
| GSN            | 92.2±7.5    | 68.2±2.5    | 76.6±5.0    | 83.5±2.0    | 77.8±3.3    | 54.3±3.3    | N/A         | 85.5±1.2    |
| SIN            | N/A         | N/A         | 76.4±3.3    | 82.7±2.1    | 75.6±3.2    | 52.4±2.9    | 92.2±1.0    | N/A         |
| CIN            | 92.7±6.1    | 68.2±5.6    | 77.0±4.3    | 83.6±1.4    | 75.6±3.7    | 52.7±3.1    | 92.4±2.1    | N/A         |
| PIN            | N/A         | N/A         | 78.8±4.4    | 85.1±1.5    | 76.6±2.9    | N/A         | N/A         | N/A         |
| N²             | N/A         | N/A         | 77.53±1.78  | 83.52±3.75  | 79.95±2.46  | 57.31±2.19  | N/A         | 86.72±1.62  |
| GMT            | 83.44±1.33  | N/A         | 75.09±0.59  | N/A         | 73.48±0.76  | 50.66±0.82  | N/A         | 80.74±0.54  |
| SEP            | 85.56±1.09  | N/A         | 76.42±0.39  | 78.35±0.33  | 74.12±0.56  | 51.53±0.65  | N/A         | 81.28±0.15  |
| Wit-TopoPool   | 93.16±4.11  | 70.57±4.43  | 80.00±3.22  | N/A         | 78.40±1.50  | 53.33±2.47  | 92.82±1.10  | N/A         |
| GrePool        | 86.25±8.35  | 59.86±6.67  | N/A         | 82.13±1.57  | N/A         | 50.77±3.25  | N/A         | 81.42±1.53  |
| RTPool         | 94.74±3.33  | 76.57±1.14  | N/A         | N/A         | 73.06±3.84  | 53.33±1.26  | N/A         | N/A         |
| **MSH-GNN (ours)** | **99.10±0.34** | **91.40±1.54** | **94.07±3.29** | **88.64±0.53** | **88.59±3.34** | **61.64±0.36** | **95.85±0.93** | **96.42±0.67** |


## Reference code
https://github.com/sunjss/N2

