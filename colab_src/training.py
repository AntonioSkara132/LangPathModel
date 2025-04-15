import torch
import torch.utils

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from LangPathModel.colab_src.nn import TrajectoryModel

from LangPathModel.src.dataset_preprocessing import CirclePathDataset, collate_fn

# Instantiate the dataset
dataset = CirclePathDataset("/content/LangPathModel/data/circle_in_the middle.pt")

# Create a DataLoader with the custom collate_fn
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

model = TrajectoryModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")


train(model = model, niter = 10, dataloader = dataloader, device = device)
