from torch.utils.data import Dataset

## Define the dataset class
# SimpleDataset: x, y
# TwoOutputDataset: x, y1, y2

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y), \
            'The number of inputs(%d) and targets(%d) does not match.' % (len(x), len(y))
        self.x = x
        self.y = y
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class TwoOutputDataset(Dataset):
    def __init__(self, x, y1, y2):
        assert len(x) == len(y1), \
            'The number of inputs(%d) and targets(y1, %d) does not match.' % (len(x), len(y1))
        assert len(x) == len(y2), \
            'The number of inputs(%d) and targets(y2, %d) does not match.' % (len(x), len(y2))
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.len = len(self.y1)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y1[index], self.y2[index]
