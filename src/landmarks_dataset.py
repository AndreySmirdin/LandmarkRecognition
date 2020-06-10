import torch
import os
from PIL import Image
from utils import transforms
from torch.utils.data import DataLoader, Dataset

class LandmarksDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.landmarks = os.listdir(root_dir)

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.landmarks[idx][1]
        image = transforms['val'](image)
        return image