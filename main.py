



from scr.preprocessing.collapse_dimensions import CollapseDimensions


if __name__ == "__main__":
    
    from scr.data.synthetic_loader import SyntheticLoader
    from scr.data.synthetic_dataset import SyntheticDataset
    from scr.preprocessing.fft import FFT
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    
    dataset = SyntheticDataset('data/synthetic_data/test_hmm')
    # Keep a handle to the transform to access window_size for plotting
    fft_transform = FFT({'window_size': 512})
    dimension_transform = CollapseDimensions({})
    dataloader = SyntheticLoader(
        dataset=dataset,
        batch_size=5120,
        shuffle=False,
        transforms=[
            fft_transform,
            dimension_transform
        ]
    )

    data = [(x,y) for x,y in dataloader]
    x = np.concatenate([xx for xx, _ in data])
    y = np.concatenate([yy for _, yy in data])

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    print(dataset.data.shape)
    print(dataset.labels.shape)
    print(dataset.raw_labels.shape)

    print(x.shape)
    print(y.shape)