"""Learnable temporal CNN front-end (replaces fixed FFT) for raw_cnn VAE pipeline."""

from __future__ import annotations

import torch.nn as nn


def conv1d_out_len(L: int, k: int, s: int, p: int, d: int = 1) -> int:
    return (L + 2 * p - d * (k - 1) - 1) // s + 1


def build_temporal_front(
    in_channels: int,
    input_length: int,
    channels: list[int],
    kernels: list[int],
    strides: list[int],
    paddings: list[int],
    pool_kernel: int | None = 2,
) -> tuple[nn.Sequential, list[int], int]:
    """(N, C_in, L_in) -> (N, C_out, L_out). Returns module, length trace, out_channels."""
    layers: list[nn.Module] = []
    lens = [input_length]
    in_ch = in_channels
    L = input_length

    for out_ch, k, s, p in zip(channels, kernels, strides, paddings):
        layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
        layers.append(nn.LeakyReLU(0.2))
        L = conv1d_out_len(L, k, s, p)
        lens.append(L)
        in_ch = out_ch

    if pool_kernel is not None and pool_kernel > 1:
        layers.append(nn.MaxPool1d(kernel_size=pool_kernel))
        L = conv1d_out_len(L, pool_kernel, pool_kernel, 0)
        lens.append(L)

    return nn.Sequential(*layers), lens, in_ch


def build_temporal_front_decoder(
    out_channels: int,
    channels: list[int],
    kernels: list[int],
    strides: list[int],
    paddings: list[int],
    pool_kernel: int | None,
    front_lens: list[int],
) -> nn.Sequential:
    """Mirror encoder front: (N, C_mid, L_mid) -> (N, out_channels, L_in)."""
    dec_layers: list[nn.Module] = []
    rev_channels = list(reversed(channels))
    rev_k = list(reversed(kernels))
    rev_s = list(reversed(strides))
    rev_p = list(reversed(paddings))

    L_in = front_lens[-1]
    in_ch = channels[-1]

    # Undo max-pool (last step in encoder front).
    conv_depth = len(channels)
    if pool_kernel is not None and pool_kernel > 1:
        target_L = front_lens[-2]
        k = pool_kernel
        s = pool_kernel
        p = 0
        base = (L_in - 1) * s - 2 * p + k
        out_pad = target_L - base
        if not (0 <= out_pad < s):
            raise ValueError(f"front decoder pool: target={target_L}, out_pad={out_pad}")
        dec_layers.append(
            nn.ConvTranspose1d(
                in_ch, in_ch, kernel_size=k, stride=s, padding=p, output_padding=out_pad
            )
        )
        dec_layers.append(nn.LeakyReLU(0.2))
        L_in = target_L

    for i in range(conv_depth):
        # front_lens: [L_in_raw, after_conv0, ..., after_conv_{n-1}, (after_pool)]
        target_L = front_lens[conv_depth - 1 - i]

        k, s, p = rev_k[i], rev_s[i], rev_p[i]
        base = (L_in - 1) * s - 2 * p + k
        out_pad = target_L - base
        if not (0 <= out_pad < s):
            raise ValueError(
                f"front decoder layer {i}: target={target_L}, base={base}, out_pad={out_pad}"
            )

        out_ch = out_channels if i == conv_depth - 1 else rev_channels[i + 1]
        dec_layers.append(
            nn.ConvTranspose1d(
                in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=out_pad
            )
        )
        if i != conv_depth - 1:
            dec_layers.append(nn.LeakyReLU(0.2))
        L_in = target_L
        in_ch = out_ch

    return nn.Sequential(*dec_layers)
