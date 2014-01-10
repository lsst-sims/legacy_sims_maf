import numpy
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.utils.testUtils as tu
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics
import glob
import numpy as np
import healpy as hp
from scipy.spatial import KDTree as kdtree




    
def treexyz(ra, dec):
    """Calculate x/y/z values for ra/dec points, ra/dec in radians."""
    # Note ra/dec can be arrays.
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z
 

def setRad(radius=1.8):
    """Set radius (in degrees) for kdtree search.

    kdtree queries will return pointings within rad."""        
    x0, y0, z0 = (1, 0, 0)
    x1, y1, z1 = treexyz(np.radians(radius), 0)
    rad = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
    return rad


bandpass = 'r'

dbTable = 'output_opsim3_61'
dbAddress = 'postgres://calibuser:calibuser@ivy.astro.washington.edu:5432/calibDB.05.05.2010'


table = db.Table(dbTable, 'obshistid', dbAddress)
simdata = table.query_columns_RecArray(constraint="filter = \'%s\'" %(bandpass), 
                                       colnames=['filter', 'expmjd',  'night',
                                                 'fieldra', 'fielddec', 'airmass',
                                                 '5sigma_modified', 'seeing',
                                                 'skybrightness_modified', 'altitude',
                                                 'hexdithra', 'hexdithdec', 'fieldid'], 
                                                 groupByCol='expmjd')


# Fixing stupid postgres case-sensitivity issues.
simdata.dtype.names = 'obsHistID', 'filter', 'expMJD', 'night', 'fieldRA', 'fieldDec', 'airmass', '5sigma_modified', 'seeing', 'skybrightness_modified', 'altitude', 'hexdithra', 'hexdithdec', 'fieldID'

# Eliminate the observations where hexdithra has failed for some reason
good=np.where((simdata['hexdithra'] < np.pi*2) )
simdata=simdata[good]

ufid, fid_ind = np.unique(simdata['fieldID'], return_index=True)

nside = 128
hpmap = np.zeros(hp.nside2npix(nside))
hpid = np.arange(hpmap.size)
hpdec,hpra = hp.pix2ang(nside, hpid)
hpdec = hpdec-np.pi/2

hpx,hpy,hpz = treexyz(hpra,hpdec)

x,y,z = treexyz(simdata['fieldRA'][fid_ind],simdata['fieldDec'][fid_ind])
tree = kdtree(zip(x,y,z), leafsize=100)

rad_max = setRad()

distances, locs = tree.query(zip(hpx,hpy,hpz), distance_upper_bound=rad_max)

good = np.where(locs < 3408)[0]
hpmap[good] = ufid[locs[good]]
hpmap[np.isinf(distances)] = hp.UNSEEN

hp.mollview(hpmap, rot=(0,0,180), unit='Field ID #')

plt.savefig('fieldID_healpix.png')

hp.mollview(hpmap, rot=(0,0,180), unit='Field ID #', cmap=plt.cm.hot)
plt.savefig('fieldID_healpix_heat.png')


rad_max = setRad(radius=1.2)
distances, locs = tree.query(zip(hpx,hpy,hpz), distance_upper_bound=rad_max)

good = np.where(locs < 3408)[0]
hpmap[good] = ufid[locs[good]]
hpmap[np.isinf(distances)] = hp.UNSEEN

hp.mollview(hpmap, rot=(0,0,180), unit='Field ID #')

plt.savefig('fieldID_healpix_smallrad.png')

