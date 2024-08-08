import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    # code from Johnson et. al. (2016)
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    np.random.seed(1)

    features = np.random.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    feats = 10 * np.einsum('ti,tij->tj', features, rotations)

    data = np.random.permutation(np.hstack([feats, labels[:, None]]))

    return data[:, 0:2], data[:, 2].astype(int)

class PinwheelDataset(Dataset):
    """
    Class for creating toy pinwheel dataset for testing VAEs.
    
    """
    def __init__(self, radial_std, tangential_std, num_classes, num_per_class, rate):
        self.data, self.labels = make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx]).long()

# Use it like this:
# dataset = PinwheelDataset(0.3, 0.05, 3, 100, 0.25)

if __name__=='__main__':
    data, labels = make_pinwheel_data(0.3, 0.05, 5, 150, 0.25)
    print(data.shape)
    figure = plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=20, cmap=plt.cm.Spectral)
    plt.xticks([])
    plt.yticks([])
    plt.show()