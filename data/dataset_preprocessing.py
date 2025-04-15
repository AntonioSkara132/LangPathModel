import torch
import torch.utils

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset, DataLoader

class CirclePathDataset(Dataset):
    def __init__(self, file_path):
        # Load the data from the .pt file
        self.data = torch.load(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the path tensor and the text for the given index
        path_tensor = self.data[idx]['path']
        text = self.data[idx]['text']
        return path_tensor, text


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Unzip the batch into paths and texts
    paths, texts = zip(*batch)
    
    # Pad the paths (ensure they're all the same length)
    padded_paths = pad_sequence(paths, batch_first=True, padding_value=0)  # Padding value can be set to 0
    
    return padded_paths, texts

# Instantiate the dataset
dataset = CirclePathDataset("/home/antonio/Workspace/Seminar/LangPathModel/data/circle_in_the middle.pt")

# Create a DataLoader with the custom collate_fn
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

for i, j in dataloader:
    print(j)
