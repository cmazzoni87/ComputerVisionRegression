import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField

#Pass timeseries and create a Gramian Angular Field image

#grab timeseries image and make the charts
def create_gaf(ts, img_size = 24):
    data = dict()
    data['gasf'] = []
    data['X_gadf'] = []
    gasf = GramianAngularField(image_size=img_size, method='summation')
    data['gasf'].append(gasf.fit_transform(ts))
    gadf = GramianAngularField(image_size=img_size, method='difference')
    data['gadf'].append(gadf.fit_transform(ts))
    return data

#create images of the bundle that we pass
def create_images(X_plots, image_name, image_matrix = (2, 2)):
    fig = plt.figure(figsize=(img * 4 for img, in image_matrix))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=image_matrix,
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.3,
                     )
    images = X_plots
    for image, ax in zip(images, grid):
        ax.imshow(image, cmap='rainbow', origin='lower')
    fig.savefig('GramianAnagularField\\{}.png'.format(image_name))

