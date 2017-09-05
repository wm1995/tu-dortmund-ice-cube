'''
Module for generating plots to assess model performance

'''
import numpy as np

# Set backend
import matplotlib
matplotlib.use('gtk')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from sklearn.metrics import precision_recall_curve

def purity_efficiency_plot(y_true, y_pred, inset=True):
    # Purity = precision
    # Efficiency = recall
    purity, efficiency, thresholds = precision_recall_curve(y_true[:, 1], y_pred[:, 1])

    # Discard last point in purity, efficiency
    # (sklearn sets them to 1, 0 respectively)
    purity = purity[:-1]
    efficiency = efficiency[:-1]

    # Code adapted from http://matplotlib.org/1.3.1/mpl_toolkits/axes_grid/examples/inset_locator_demo.py
    fig, ax = plt.subplots()

    ax.plot(thresholds, purity, label="Purity")
    ax.plot(thresholds, efficiency, label="Efficiency")

    axins = zoomed_inset_axes(ax, zoom=5, loc=4, borderpad=5)

    axins.plot(purity, thresholds)
    axins.plot(efficiency, thresholds)

    axins.set_xlim(0.9, 1)
    axins.set_ylim(0.9, 1)

    ax.xlabel("Confidence Cut")
    ax.ylabel("Quality Parameter")

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.legend(loc=2)

    plt.draw()
    plt.show()