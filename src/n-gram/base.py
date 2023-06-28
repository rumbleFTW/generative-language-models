import torch

### --- Globals
checkpt_path = "../../checkpts/ngram.pt"
index_path = "../../index_tables/ngram.json"
### --- Hyperparameters
epochs = 10
n = 8
split_ratio = 0.8
output_tokens = 30
### ---

device = "cuda" if torch.cuda.is_available() else "cpu"
