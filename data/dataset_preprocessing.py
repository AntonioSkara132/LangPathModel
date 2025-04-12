import torch
import torch.utils

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    paths, texts = zip(*batch)
    lengths = torch.tensor([p.size(0) for p in paths])
    padded_paths = pad_sequence(paths, batch_first=True)
    return padded_paths, texts, lengths


class PathDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.data = torch.load(file_path)  # List of [seq_len, 4] tensors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = PathDataset("data/circle_in_the middle.pt")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
"""
for batch_paths, batch_texts, lengths in dataloader:
    print("Paths shape:", batch_paths.shape)  # [B, T, 4]
    print("Text conditions:", batch_texts)    # list of strings
    print("Lengths:", lengths)                # original lengths
"""
it = iter(dataloader)

print(next(it))