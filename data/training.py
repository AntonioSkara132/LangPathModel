import torch
import torch.utils

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from trainer import train
from nn import TrajectoryModel

from dataset_preprocessing import CirclePathDataset, collate_fn

# Instantiate the dataset
dataset = CirclePathDataset("/home/antonio/Workspace/Seminar/LangPathModel/data/circle_in_the middle.pt")

# Create a DataLoader with the custom collate_fn
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

model = TrajectoryModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(model = model, niter = 1, dataloader = dataloader, device = device)




