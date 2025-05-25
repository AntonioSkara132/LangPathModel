import torch
import torch.utils

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset

class PathDataset(Dataset):
    """
    Loads paths and captions from a .pt file of the form:
        [{"path": Tensor[N,3], "text": str}, ...]
    Normalises XY coordinates to roughly [-1, 1].
    """
    def __init__(self, file_path):
        super().__init__()
        self.data = torch.load(file_path, map_location="cpu")  # robust to GPU-saved files
        if not isinstance(self.data, (list, tuple)):
            raise TypeError(
                f"Expected list/tuple in {file_path}, got {type(self.data).__name__}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if not {"path", "text"} <= sample.keys():
            raise KeyError("Each item must contain 'path' and 'text' keys.")

        path = sample["path"].clone()                          # (N, 3)
        pos = path[:, :2]                                      # XY columns

        ## normalise per-sample, not per-point
        #max_vals = pos.abs().max(dim=0).values                 # (2,)
        #mean_vals = pos.mean(dim=0)                            # (2,)

        #denom = max_vals.clamp_min(1e-8)                       # avoid /0
        path[:, :2] /= 1000

        return path, sample["text"]



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
