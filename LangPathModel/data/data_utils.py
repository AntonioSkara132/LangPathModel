import torch
import torch.utils

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class PathDataset(Dataset):
""" loads paths and captions from .pt file"""
    def __init__(self, file_path):
        # Load the data from the .pt file
        self.data = torch.load(file_path)
        #normalization
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the path tensor and the text for the given index
        path_tensor = self.data[idx]['path'].clone()
        pos_s = path_tensor[:, :2]                         # Select first 2 elements along dim 0
        max, mean = pos_s.max(-1).values, pos_s.mean(-1)  # Max & mean over last dim
        path_tensor[:, 0:2] = (path_tensor[:, 0:2] - mean.unsqueeze(-1)) / max.unsqueeze(-1)
        text = self.data[idx]['text']
        return path_tensor, text


from torch.nn.utils.rnn import pad_sequence
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def collate_fn(batch):
""" extracts paths text and masks from batch """
    from torch.nn.utils.rnn import pad_sequence
    paths, texts = zip(*batch)

    path_lengths = torch.tensor([p.size(0) for p in paths]) 

    padded_paths = pad_sequence(paths, batch_first=True, padding_value=0)

    max_len = padded_paths.size(1)
    path_masks = torch.arange(max_len)[None, :].to(path_lengths.device) < path_lengths[:, None]  # shape: [B, T]

    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    return padded_paths, encoded, path_masks
