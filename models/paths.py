import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import hamming_loss
import blackbox_backprop as bb
import numpy as np

class HammingLoss(nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()

def main():
    data = torch.tensor(np.random.randn(100, 16, 32))
    labels = torch.tensor(np.random.randint(0, 2, size=(100, 16, 32)))
    linear_size = data.shape[1] * data.shape[2]
    dl = nn.Linear(linear_size, linear_size)
    flattened = torch.flatten(data, start_dim=1)
    shortest_paths = bb.ShortestPath.apply
    loss_fn = HammingLoss()
    opt = optim.SGD(dl.parameters(), lr=0.001)
    for i in range(20):
        opt.zero_grad()
        dl_out = dl(flattened.float())
        weights = dl_out.reshape(data.shape)
        paths_out = shortest_paths(weights, 5.0)
        loss = loss_fn(paths_out, labels)
        loss.backward()
        opt.step()
        print(loss)

if __name__ == '__main__':
    main()