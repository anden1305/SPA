



from scr.preprocessing.collapse_dimensions import CollapseDimensions


if __name__ == "__main__":
    
    from scr.data.synthetic_loader import SyntheticLoader
    from scr.data.synthetic_dataset import SyntheticDataset
    from scr.preprocessing.fft import FFT
    import numpy as np
    import matplotlib.pyplot as plt
    
    dataset = SyntheticDataset('data/synthetic_data/test')
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

    # Accumulate mean feature space (magnitude of FFT) per class across all batches
    sums_by_class: dict = {}
    counts_by_class: dict = {}
    
    # Iterate all batches and aggregate
    for X, Y in dataloader:
        # X: (n, C*F) complex after rFFT then flattened; Y: (n,) labels per window (majority voting)
        mag_flat = np.abs(X)  # convert to magnitudes for plotting

        # Recover (C, F) from flattened dimension using dataset channels and FFT bins
        C = int(dataset.data.shape[0])
        assert mag_flat.shape[1] % C == 0, "Flattened feature size is not divisible by channels."
        F = mag_flat.shape[1] // C
        # Optional sanity check against rFFT bin count
        F_expected = fft_transform.window_size // 2 + 1
        assert F == F_expected, f"Expected F={F_expected}, got F={F}."

        mag = mag_flat.reshape(-1, C, F)  # (n, C, F)

        # Aggregate by class present in this batch
        for cls in np.unique(Y):
            mask = (Y == cls)
            if not np.any(mask):
                continue
            batch_sum = mag[mask].sum(axis=0)  # (C, F)
            if cls not in sums_by_class:
                sums_by_class[cls] = batch_sum
                counts_by_class[cls] = int(mask.sum())
            else:
                sums_by_class[cls] += batch_sum
                counts_by_class[cls] += int(mask.sum())

    # Compute mean spectra per class
    means_by_class = {
        cls: sums_by_class[cls] / max(counts_by_class[cls], 1)
        for cls in sums_by_class
    }

    # Frequency axis in Hz
    freqs = np.fft.rfftfreq(fft_transform.window_size, d=1 / dataset.sampling_rate)

    # Plot mean across channels per class
    plt.figure(figsize=(10, 6))
    for cls in sorted(means_by_class.keys(), key=lambda x: str(x)):
        mean_C_F = means_by_class[cls]  # (C, F)
        mean_F = mean_C_F.mean(axis=0)  # average over channels
        plt.plot(freqs, mean_F, label=f"{cls} (n={counts_by_class[cls]})")

    plt.title("Mean FFT Magnitude per Class (channel-averaged)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()