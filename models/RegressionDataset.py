from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
