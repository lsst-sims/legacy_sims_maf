import numpy as np
import healpy as hp
import matplotlib.pylab as plt

import glob

files = glob.glob('Chips/*.npz')

for fname in files:
    data = np.load(fname)

    mapv = data['metricValues']
    mapv[data['mask']] = hp.UNSEEN

    cbarFormat='%i'

    hp.gnomview(mapv, rot=(5.5,-5,0), xsize=300, ysize=300, title='',cbar=False)

    ax = plt.gca()
    im = ax.get_images()[0]
    cb = plt.colorbar(im, shrink=0.75, aspect=25, orientation='horizontal',
                      extend='both', extendrect=True, format=cbarFormat)

    cb.set_label('Number of Observations')

    cb.solids.set_edgecolor("face")

    plt.savefig(fname[:-4]+'_reproject.pdf')
