################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
################################################################################

import torch
import numpy as np
from typing import List
import matplotlib.pyplot as plt


def complex_mse_loss(output, target, norm=True):
    """ Computes the mean squared error between two arrays. """
    diff = output - target
    den = 1. if not norm else np.linalg.norm((target * np.conj(target)).mean())
    mse = np.linalg.norm((diff * np.conj(diff)).mean()) / den
    return mse


def db(v):
    """ Converts a linear power to dB. """
    return 10 * np.log10(v)

# Plotting


def plot_losses(loss_list, loss_labels=[], title=None, savename=''):
    """
    Plot model losses. The loss list can have multiple arrays/lists for each
    model/estimator/strategy.
    """
    if not loss_labels:
        loss_labels = [f'loss {i}' for i in range(len(loss_list))]

    plt.figure(dpi=200)
    for i, loss in enumerate(loss_list):
        plt.plot(loss, label=loss_labels[i])
    if title:
        plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Complex MSE loss [dB]')
    plt.legend()
    plt.grid()
    if savename:
        plt.savefig(savename)
    plt.show()


def compare_ch_ests(ys: List[np.ndarray], labels: List[str] = None,
                    title: str = None, save_path: str = None) -> None:
    """
    Plots the magnitude and phase of each channel estimate.
    """
    n_est = len(ys)

    if not labels:
        labels = [f'est{i}' for i in range(n_est)]

    n_subcarriers = max((ys[i].shape[-1] for i in range(n_est)))
    interleave = [n_subcarriers // ys[i].shape[-1] for i in range(n_est)]

    fig, axs = plt.subplots(2, 1, dpi=200, tight_layout=True, figsize=(8, 6))
    fs = [np.abs, np.angle]  # magnitude in 1st row, phases in 2nd row

    subcarriers = np.arange(n_subcarriers)
    for i in range(2):
        for est_i in range(n_est):
            axs[i].plot(subcarriers[::interleave[est_i]], fs[i](ys[est_i]), label=labels[est_i])
        axs[i].legend(loc='upper right')
        axs[i].set_xlabel('Subcarrier')
        axs[i].set_ylabel(['Magnitude (dB)', 'Phase (radians)'][i])
        axs[i].grid()
    if title:
        fig.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_annotaded_cdfs(ys: List[np.ndarray], labels: List[str] = None, title: str = ''):
    """
    Compute and plot CDFs for each array in the list, and marke mean and median.
    Used mainly to show the CDFs of MSE losses for N realizations of a given
    channel estimation approach.
    """

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    labels = [f'y_{i}' for i in len(ys)] if not labels else labels

    plt.figure(dpi=150)
    plt.grid()
    plt.xlabel('MSE (dB)')
    plt.ylabel('CDF')

    for y_idx, y in enumerate(ys):
        c = colors[y_idx]
        lab = labels[y_idx]
        mean, median = db(np.mean(y)), db(np.median(y))

        # Plot 2: Cumulative distributions of MSEs in dB
        data_sorted_db = db(np.sort(y))

        cdf = np.arange(1, len(data_sorted_db) + 1) / len(data_sorted_db)

        # CDFs
        plt.plot(data_sorted_db, cdf, color=c, lw=2, label=lab + ' CDF')

        # Mark median
        plt.scatter(median, 0.5, color=c, label=lab + ' mean')

        # Mark mean
        x_mean = cdf[data_sorted_db > mean][0]
        plt.scatter(mean, x_mean, marker='s', color=c, label=lab + ' median')

    plt.legend(ncols=len(ys))
    plt.title(title)
    plt.show()


# Path Management


def get_snr_model_path(model_dir, snr):
    """ Return a path where a model trained for a specific SNR will be saved. """
    return model_dir + f'model_SNR={snr:.1f}.path'


def get_model_training_dir(folder, channel, n_prb, n_iter, batch_size):
    """ Return a path to a directory/folder where several models will be saved. """
    path = (f'{folder}/ch={channel}_n_iter={n_iter}_'
            f'batchsize={batch_size}_model={n_prb}PRBs/')
    return path


# Torch Utils


def roll_in_half(x):
    """
    Rolls last dimension of array x by half of its size
    E.g.  [1,2,3,4,5,6] -> [4,5,6,1,2,3]
    """
    n_last_dim = x.shape[-1]
    return torch.roll(x, int(n_last_dim / 2.), dims=-1)


def unroll_in_half(x):
    """
    Unrolls last dimension of array x by half of its size
    E.g.  [4,5,6,1,2,3] -> [1,2,3,4,5,6]
    """
    n_last_dim = x.shape[-1]
    return torch.roll(x, -int(n_last_dim / 2.), dims=-1)


def sfft(x, roll=True):
    """ Compute FFT with option to roll in half afterwards."""
    y = torch.fft.fft(x)
    if roll:
        y = roll_in_half(y)
    return y


def isfft(x, roll=True):
    """ Compute IFFT with option to unroll in half beforehand."""
    if roll:
        x = unroll_in_half(x)
    return torch.fft.ifft(x)
