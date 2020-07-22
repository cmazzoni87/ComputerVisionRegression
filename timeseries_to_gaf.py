import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid, AxesGrid
from pyts.image import GramianAngularField
import pandas as pd

#Pass timeseries and create a Gramian Angular Field image

#grab timeseries image and make the charts
def create_gaf(ts): #col_num
    data = dict()
    # gasf = GramianAngularField(method='summation', image_size=20)
    # data['gasf'] = gasf.fit_transform(pd.DataFrame(ts).T)[0]  # ts.T)
    gadf = GramianAngularField(method='difference', image_size=20)
    data['gadf'] = gadf.fit_transform(pd.DataFrame(ts).T)[0] # ts.T)
    # create_images([data['gasf'][0], data['gasf'][0], data['gasf'][0], data['gasf'][0]], 'testo')
    return data

#create images of the bundle that we pass
def create_images(X_plots, image_name, destination, image_matrix=(2, 2)):
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

    fig.savefig('GramianAnagularFields\\{}.png'.format(destination + '\\' + image_name))
    plt.close(fig)

