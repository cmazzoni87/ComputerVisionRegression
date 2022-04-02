import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
import pandas as pd
import numpy as np
import os
from typing import *

matplotlib.use('Agg')


# Pass times-eries and create a Gramian Angular Field image
# Grab times-eries and draw the charts
def create_gaf(ts) -> Dict[str, Any]:
    """
    :param ts:
    :return:
    """
    data = dict()
    gadf = GramianAngularField(method='difference', image_size=ts.shape[0])
    data['gadf'] = gadf.fit_transform(pd.DataFrame(ts).T)[0]
    return data


# Create images of the bundle that we pass
def create_images(X_plots: Any, image_name: str, destination: str) -> None:
    """
    :param X_plots:
    :param image_name:
    :param destination:
    :return:
    """
    
    window_size = np.sqrt(len(X_plots))
    assert window_size % 1 == 0. and window_size > 1, "Non-square GAF array not supported! (i.e. 2x2, 3x3 etc. is required)"
    window_size = int(window_size)
    if window_size > 2:
        # Figure size like 3x3, 4x4, 5x5 etc.
        # Get top and bottom tiles
        top = np.hstack((X_plots[0:window_size]))
        bot = np.hstack((X_plots[-window_size:]))

        # Get middle tiles horizontally
        midTmp = []
        for i in range(window_size - 2):
            midTmp.append(np.hstack((X_plots[(i + 1)*window_size:(i + 1)*window_size + window_size])))

        # Stack middle tiles vertically
        for j, val in enumerate(midTmp):
            if j == 0:
                mid = val
            else:
                mid = np.vstack((mid,val))

        # Final assembly
        full = np.vstack((top,mid,bot))
    else:
        # Simple case for 2x2 
        top = np.hstack((X_plots[0], X_plots[1]))
        bot = np.hstack((X_plots[2], X_plots[3]))
        full = np.vstack((top, bot))

    repo = os.path.join('GramianAngularFields/TRAIN', destination)
    plt.imsave(os.path.join(repo, image_name + ".png"), full, cmap="rainbow")