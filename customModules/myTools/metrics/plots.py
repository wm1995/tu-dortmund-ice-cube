'''
Module for generating plots to assess model performance

'''
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from sklearn.metrics import precision_recall_curve

def purity_efficiency_plot(y_true, y_pred, inset=True, savepath=None):
    # Purity = precision
    # Efficiency = recall
    purity, efficiency, thresholds = precision_recall_curve(y_true[:, 1], y_pred[:, 1])

    # Discard last point in purity, efficiency
    # (sklearn sets them to 1, 0 respectively)
    purity = purity[:-1]
    efficiency = efficiency[:-1]

    # Set larger font for legibility
    matplotlib.rcParams.update({'font.size': 14})

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