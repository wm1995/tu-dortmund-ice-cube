'''
Module for generating plots to assess model performance

'''
from __future__ import absolute_import
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from sklearn.metrics import precision_recall_curve

FONT_SIZE = 14

def purity_efficiency_plot(y_true, y_pred, inset=True, savepath=None):
    # Purity = precision
    # Efficiency = recall
    purity, efficiency, thresholds = precision_recall_curve(y_true[:, 1], y_pred[:, 1])

    # Discard last point in purity, efficiency
    # (sklearn sets them to 1, 0 respectively)
    purity = purity[:-1]
    efficiency = efficiency[:-1]

    # Set larger font for legibility
    matplotlib.rcParams.update({'font.size': FONT_SIZE})

    # Code adapted from http://matplotlib.org/1.3.1/mpl_toolkits/axes_grid/examples/inset_locator_demo.py
    fig, ax = plt.subplots()

    # Plot purity and efficiency
    ax.plot(thresholds, purity, label="Purity")
    ax.plot(thresholds, efficiency, label="Efficiency")

    # If necessary, make an inset plot
    if inset:
        # Use a zoom of five, place in lower left corner, pad border by 3
        axins = zoomed_inset_axes(ax, zoom=5, loc=4, borderpad=3)

        axins.plot(purity, thresholds)
        axins.plot(efficiency, thresholds)

        # Set limits for inset plot
        axins.set_xlim(0.9, 1)
        axins.set_ylim(0.9, 1)
	
	# Set ticks for inset plot
	axins.set_xticks([0.9, 0.95, 1.0])
        axins.set_yticks([0.9, 0.95, 1.0])

        # Mark inset box
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_xlabel("Confidence Cut")
    ax.set_ylabel("Quality Parameter")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax.legend(bbox_to_anchor=(0.5, 1), loc="lower center", ncol=2, frameon=False)

    if savepath == None:
        plt.show()
    else:
        plt.savefig(savepath)

def rate_plot(y_true, y_pred, weights, bin_size=0.02, combine_nu_tau_cc=False, savepath=None):
    bins = np.arange(0, 1.0, bin_size)
    counts = np.zeros((y_true.shape[1], len(bins) + 1))
    inds = np.digitize(y_pred[:, 1], bins)
    label_mask = y_true.astype(bool)

    # Set larger font for legibility
    matplotlib.rcParams.update({'font.size': FONT_SIZE})

    for i in range(len(inds)):
        counts[label_mask[i, :], inds[i]] += weights[i]

    labels = {'0':'Cascade-like background',
              '1':'Double-pulse events',
              '2':'Other nu_tau cc events',
              '3':'nu_mu cc events',
              '4':'Atmospheric muons'}

    if combine_nu_tau_cc:
        labels['1'] ='nu_tau cc events'

    bin_range = np.arange(0, 1 + 1e-8, stepsize)

    fig, ax = plt.subplots()

    box = ax.get_position()
    x_offset = 0.03
    if counts.shape[0] == 2:
        y_offset = 0
    elif counts.shape[0] == 5:
        y_offset = 0.05

    ax.set_position([box.x0 + box.width*x_offset, box.y0, box.width * (1 - x_offset), box.height * (1 - y_offset)])

    for i in range(counts.shape[0]):    
        if combine_nu_tau_cc:
            if i == 1 and counts.shape[0] > 2:
                counts[1] += counts[2]
            elif i == 2:
                continue
        ax.step(bin_range, counts[i], label=labels[str(i)])

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Waveforms per year")
    ax.set_xlim(0, 1)
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(0.5, 1), loc="lower center", ncol=2, frameon=False)

    if savepath == None:
        plt.show()
    else:
        plt.savefig(savepath)
