

from scr.data.synthetic_dataset import SyntheticDataset
from scr.data.mssv_dataset import MSSVDataset
from scr.data.data_loader import DataLoader

sd = SyntheticDataset(id="test")
dl = DataLoader(dataset=sd, batch_size=5120, transforms=[])

for epoch in range(10):
    for i, (x, y) in enumerate(dl):
        print(f"Epoch {epoch}, Batch {i}, Batch shape: {x.shape}, Labels shape: {y.shape}")