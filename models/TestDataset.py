from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, x, transform=None):
        self.x = x
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]

        if self.transform:
            img = self.transform(img)

        return img