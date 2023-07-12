import pathlib
import torch

import numpy as np

class ExploreDataset(torch.utils.data.Dataset):
    
    def __init__(self, root = "."):
        if isinstance(root, str):
            root = pathlib.Path(root)
        self.lengths = np.load(root / "lengths.npy")
        self.xs = np.load(root / "x_padded.npy")
        self.ys = np.load(root / "y_padded.npy")

    def __len__(self):
      return len(self.lengths)

    def __getitem__(self, index):
      return self.xs[index], self.ys[index], self.lengths[index]

    @staticmethod
    def numpy_collate(batch):
       if isinstance(batch[0], np.ndarray):
          return np.stack(batch)
       elif isinstance(batch[0], (tuple,list)):
         transposed = zip(*batch)
         return [ExploreDataset.numpy_collate(samples) for samples in transposed]
       else:
         return np.array(batch)