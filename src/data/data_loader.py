
from typing import Iterator
import torch
import numpy as np
from src.data.base_dataset import BaseDataset
from src.config.config import GlobalConfig, TransformsConfig
from src.preprocessing.beta_delta_pac import BetaDeltaPAC
from src.preprocessing.beta_delta_ratio import BetaDeltaRatio
from src.preprocessing.burst_rate import BurstRate
from src.preprocessing.envelope_gini import EnvelopeGini
from src.preprocessing.theta_beta_pac import ThetaBetaPAC
from src.preprocessing.theta_beta_ratio import ThetaBetaRatio
from src.preprocessing.theta_delta_pac import ThetaDeltaPAC
from src.preprocessing.theta_gamma_pac import ThetaGammaPAC
from src.preprocessing.duty_cycle import DutyCycle
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.vae_preprocessing import VAEPreprocessing
from src.preprocessing.fft_band_power import FFTBandPower
from src.preprocessing.fft_log_power import FFTLogPower
from src.preprocessing.fft_power import FFTPower
from src.preprocessing.fft_relative_band_power import FFTRelativeBandPower
from src.preprocessing.band_pass_filter import BandPassFilter
from src.preprocessing.theta_peak_quality import ThetaPeakQuality
from src.preprocessing.percentile_clipping import PercentileClipping
from src.preprocessing.absolute_power import AbsolutePower
from src.preprocessing.rms import RMS
from src.preprocessing.theta_delta_ratio import ThetaDeltaRatio
from src.preprocessing.theta_to_beta_gamma_ratio import ThetaToBetaGammaRatio

class DataLoader(Iterator):
    """Documentation

    Abstract class for all data loaders that are needed for this codebase.
    """
    
    def __init__(self, 
                 dataset: BaseDataset, 
                 config: GlobalConfig,
                 device: torch.device = torch.device("cpu")):
        self.global_config = config
        self.config = self.global_config.dataloader
        self.dataset = dataset
        self.batch_size = self.config.batch_size
        self.seed = self.global_config.seed
        self.shuffle = self.config.shuffle
        self.normalize = self.config.normalize
        self.device = device
        self.verbose = self.global_config.verbose
        self._epoch = 0
        self.has_features: bool = False
        
        # legacy behavior
        self.use_legacy = self.config.use_legacy
        if self.use_legacy:
            self.__init_transforms(transform_configs=self.config.transforms)
            self.__process_data_legacy()
        else:
            self.process_data()
    
    def process_data(self):
        # get data
        x, y = self.dataset[:] # X (C, T), y (T,)
        
        # check that config is correct
        assert self.config.stride is not None, "stride must be set for non-legacy data loading."
        assert self.config.window_size is not None, "window_size must be set for non-legacy data loading."
        assert self.config.sequence_length is not None, "sequence_length must be set for non-legacy data loading."
        
        # create preprocessing object
        vae_preprocessing = VAEPreprocessing(
            self.global_config,
            window_size=self.config.window_size,
            stride=self.config.stride,
            sequence_length=self.config.sequence_length,
            sampling_rate=self.dataset.get_sampling_rate()
        )
        # save transforms
        self.transforms = {'EEG': [vae_preprocessing], 'EMG': [vae_preprocessing]}
        
        # apply transforms
        x, y = vae_preprocessing(x, y)
        
        # save & print data
        self.data = (x, y)
        self.__print_data_info()
    
    def __process_data_legacy(self):
        # get data
        x, y = self.dataset[:]
        x, y = self.__apply_transforms(x, y)
        # flip axis 0 and 1 if data is raw
        if x.shape[0] == 1:
            x = x.transpose(1, 0, 2)
        # normalize data
        if self.normalize:
            x = (x - x.mean(axis=(0,1))) / x.std(axis=(0,1))
        # set batch size
        if self.batch_size is None:
            batch_size = x.shape[0]
        else:
            batch_size = self.batch_size
        num_full_batches = x.shape[0] // batch_size
        if num_full_batches == 0:
            raise ValueError(f"Dataset size {x.shape[0]} is smaller than batch size {batch_size}.")
        # trim data to have full batches only
        x = x[:num_full_batches * batch_size]
        y = y[:num_full_batches * batch_size * x.shape[1]]
        # reshape data so that each batch is a continuous segment of the original data
        x = x.reshape((num_full_batches, batch_size*x.shape[1], x.shape[2]))
        y = y.reshape((num_full_batches, y.shape[0] // num_full_batches))
        # make data contiguous in memory
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        # store data
        self.data = (x, y)
        self.__print_data_info()

    def __init_transforms(self, transform_configs: list[TransformsConfig]):
        # keep transforms keyed by channel name in upper case to match dataset channel labels
        self.transforms: dict[str, list[BaseTransform]] = {"EEG": [], "EMG": []}
        for config in transform_configs:
            channel = config.channel.upper()
            if channel not in self.transforms:
                raise ValueError(f"Unknown channel for transform: {config.channel}. Expected one of {list(self.transforms.keys())}.")
            ttype = config.type.lower()
            match ttype:
                case "fft_power":
                    self.transforms[channel].append(
                        FFTPower(
                            config=config, 
                            window_size=self.config.window_size, 
                            stride=self.config.stride, 
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case "fft_log_power":
                    self.transforms[channel].append(
                        FFTLogPower(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case "fft_band_power":
                    self.transforms[channel].append(
                        FFTBandPower(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'fft_relative_band_power':
                    self.transforms[channel].append(
                        FFTRelativeBandPower(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'theta_delta_ratio':
                    self.transforms[channel].append(
                        ThetaDeltaRatio(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'beta_delta_ratio':
                    self.transforms[channel].append(
                        BetaDeltaRatio(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'theta_beta_ratio':
                    self.transforms[channel].append(
                        ThetaBetaRatio(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'absolute_power':
                    self.transforms[channel].append(
                        AbsolutePower(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case "rms":
                    self.transforms[channel].append(
                        RMS(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                        )
                    )
                case "theta_peak_quality":
                    self.transforms[channel].append(
                        ThetaPeakQuality(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'duty_cycle':
                    self.transforms[channel].append(
                        DutyCycle(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                        )
                    )
                case 'theta_to_beta_gamma_ratio':
                    self.transforms[channel].append(
                        ThetaToBetaGammaRatio(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'theta_gamma_pac':
                    self.transforms[channel].append(
                        ThetaGammaPAC(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'theta_delta_pac':
                    self.transforms[channel].append(
                        ThetaDeltaPAC(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'theta_beta_pac':
                    self.transforms[channel].append(
                        ThetaBetaPAC(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'beta_delta_pac':
                    self.transforms[channel].append(
                        BetaDeltaPAC(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'burst_rate':
                    self.transforms[channel].append(
                        BurstRate(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case 'envelope_gini':
                    self.transforms[channel].append(
                        EnvelopeGini(
                            config=config,
                            window_size=self.config.window_size,
                            stride=self.config.stride,
                            sampling_rate=self.dataset.get_sampling_rate()
                        )
                    )
                case "percentile_clipping":
                    self.transforms[channel].append(PercentileClipping(config=config))
                case "band_pass_filter":
                    self.transforms[channel].append(BandPassFilter(config=config, sampling_rate=self.dataset.get_sampling_rate()))
                case _:
                    raise ValueError(f"Unknown transform: {config.type}.")
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.data[0].shape[0]
    
    def __apply_transforms(self, x: np.ndarray, y: np.ndarray):
        """Apply transforms per channel group and combine results.

        Strategy:
        - `x` starts as shape (C, T) (channels, time) and `y` is (T,).
        - For each channel type (e.g. 'EEG', 'EMG') gather the indices in the dataset
          that match that type and build a sub-array of shape (T, C_group).
        - Apply all preprocessing transforms for that channel (in order), then all
          postprocessing transforms (in order). Each transform receives and returns
          (x, y) where x is typically (T, C) or (N, T, C). We normalize to a
          3D array (N, T, C_group_processed) for later concatenation.
        - Finally concatenate processed channel groups along the feature/channel
          axis (last axis) and return (N, T, C_total) together with the labels.
        """
        
        # initial data is (C, T)
        channels_meta = self.dataset.get_channels()

        processed_groups: list[np.ndarray] = []
        processed_y = None
        feature_names = []
        for ch_type, transforms in self.transforms.items():
            
            # find indices for this channel type (case-insensitive)
            idxs = [i for i, ch in enumerate(channels_meta) if ch.upper() == ch_type.upper()]
            if len(idxs) == 0:
                # nothing to do for this channel type
                continue
            
            # extract group data: result shape (n_ch, T)
            group_x = x[idxs, :]
            # convert to (T, C_group) which most transforms expect
            group_x = group_x.transpose(1, 0)
            group_y = y
            
            # split transforms by stage
            pre_transforms = [t for t in transforms if t.get_stage() == "preprocessing"]
            post_transforms = [t for t in transforms if t.get_stage() == "postprocessing"]

            # apply preprocessing
            for t in pre_transforms:
                group_x, group_y = t(group_x, group_y)

            # apply postprocessing
            group_x_features = []
            group_y_features = []
            for t in post_transforms:
                _group_x, _group_y = t(group_x, group_y)
                for channel_index in range(_group_x.shape[2]):
                    for feature_index in range(_group_x.shape[1]):
                        feature_names.append(f"{t.get_short_name()}{' ' + str(feature_index+1) if _group_x.shape[1] > 1 else ''} ({ch_type}{' ' + str(channel_index+1) if _group_x.shape[2] > 1 else ''})")
                if _group_x.shape[1] > 1:
                     _group_x = _group_x.reshape(_group_x.shape[0], 1, _group_x.shape[1] * _group_x.shape[2])
                group_x_features.append(_group_x)
                group_y_features.append(_group_y)
            
            # concatenate postprocessed features along last axis
            if len(group_x_features) > 0:
                group_x = np.concatenate(group_x_features, axis=2)
                self.has_features = True
                # for labels, ensure consistency across postprocessing transforms
                first_y = group_y_features[0]
                for other_y in group_y_features[1:]:
                    if not np.array_equal(first_y, other_y):
                        raise ValueError("Labels produced by postprocessing transforms are inconsistent.")
                group_y = first_y

            if group_x.ndim == 2:
                group_x = np.expand_dims(group_x, axis=0)

            # store labels, ensuring consistency across channel groups
            if processed_y is None:
                processed_y = group_y
            else:
                if not np.array_equal(processed_y, group_y):
                    raise ValueError("Labels produced by transforms are inconsistent across channel groups.")

            processed_groups.append(group_x)

        # if no transforms were defined/applied, convert raw data to (1, T, C)
        if len(processed_groups) == 0:
            x_out = x.transpose(1, 0)
            x_out = np.expand_dims(x_out, axis=0)
            return x_out, y

        # ensure all processed groups have matching sample/time dims (N, T)
        Ns = [g.shape[0] for g in processed_groups]
        Ts = [g.shape[1] for g in processed_groups]
        if len(set(Ns)) > 1 or len(set(Ts)) > 1:
            raise ValueError(f"Transformed channel groups have mismatched shapes: Ns={Ns}, Ts={Ts}.")

        # save feature names
        self.feature_names = feature_names

        # concatenate along last axis (features / channels)
        x_out = np.concatenate(processed_groups, axis=2)
        if x_out.shape[2] > 50:
            self.has_features = False
        return x_out, processed_y
    
    def has_posttransforms(self) -> bool:
        for channel in self.transforms:
            for t in self.transforms[channel]:
                if t.get_stage() == "postprocessing":
                    return True
        return False
    
    def get_feature_names(self) -> list[str]:
        return self.feature_names
    
    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data
    
    def __iter__(self) -> "DataLoader":
        """Handles every start of new epoch logic (shuffle etc.)."""
        self._size = self.data[0].shape[0]
        self._num_batches = self._size
        batch_order = [i for i in range(self._num_batches)]
        if self.shuffle:
            rs = np.random.RandomState(self.seed + self._epoch)
            batch_order = rs.permutation(batch_order).tolist()
        self._batch_order = batch_order
        self._batch_cursor = 0
        self._epoch += 1
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Loads, transforms and returns the next batch of data."""
        if self._batch_cursor >= getattr(self, "_num_batches", 0):
            raise StopIteration
        x = self.data[0][self._batch_order[self._batch_cursor]]
        y = self.data[1][self._batch_order[self._batch_cursor]]
        x = np.expand_dims(x, axis=0)
        self._batch_cursor += 1
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
    
    def get_feature_dim(self) -> int:
        return self.data[0].shape[2]
    
    def get_all_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.data
        y = y.reshape(-1)
        x = x.reshape(-1, x.shape[2])
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        # return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
        if self.device.type == "cuda":
            xt = torch.from_numpy(x).pin_memory().to(self.device, non_blocking=True)
            yt = torch.from_numpy(y).pin_memory().to(self.device, non_blocking=True)
            return xt, yt
        else:
            return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
    
    def has_features_enabled(self) -> bool:
        return self.has_features

    def __print_data_info(self):
        print("\n" + "=" * 60)
        print("📊 Data Info".center(60, "="))
        print("=" * 60)
        print(f"Dataset: {self.dataset}")
        print(f"Number of samples: {self.data[0].shape[0] * self.data[0].shape[1]}")
        print(f"Number of batches per epoch: {self.__len__()}")
        print(f"Batch size: {self.batch_size}")
        print(f"Feature dimension: {self.get_feature_dim()}")
        print(f"Shuffle each epoch: {self.shuffle}")
        print(f"Normalize data: {self.normalize}")
        print(f"Device: {self.device}")
        print(f"Final X shape: {self.data[0].shape}")
        print(f"Final Y shape: {self.data[1].shape}")
        if self.transforms:
            print("Transforms:")
            for channel in self.transforms:
                print(f" - {channel}: {self.transforms[channel]}")
        else:
            print("No transforms applied.")
        print("=" * 60 + "\n")
    
    def __str__(self) -> str:
        return f"DataLoader(transform={self.transforms}, dataset={self.dataset})"