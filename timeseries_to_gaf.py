import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
import pandas as pd
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
def create_images(X_plots: Any, image_name: str, destination: str, image_matrix: tuple =(2, 2)) -> None:
    """
    :param X_plots:
    :param image_name:
    :param destination:
    :param image_matrix:
    :return:
    """
    fig = plt.figure(figsize=[img * 4 for img in image_matrix])
    grid = ImageGrid(fig,
                     111,
                     axes_pad=0,
                     nrows_ncols=image_matrix,
                     share_all=True,
                     )
    images = X_plots
    for image, ax in zip(images, grid):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image, cmap='rainbow', origin='lower')

    repo = os.path.join('GramianAngularFields/TRAIN', destination)
    fig.savefig(os.path.join(repo, image_name))
    plt.close(fig)
