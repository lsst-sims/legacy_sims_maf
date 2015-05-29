import numpy as np
from lsst.sims.catalogs.generation.db import ObservationMetaData
from lsst.sims.coordUtils import CameraCoords
from lsst.obs.lsstSim import LsstSimMapper
import matplotlib.pylab as plt


mapper = LsstSimMapper()
camera = mapper.camera

epoch = 2000.0

# Generate a dummy obs_metadata:
obs_metadata = ObservationMetaData(m5=0., bandpassName='g')
obs_metadata.unrefractedRA = np.pi/2.
obs_metadata.unrefractedDec = 0.
obs_metadata.rotSkyPos = 0.
obs_metadata.mjd = 49353.032079

myCamCoords = CameraCoords()

#generate a grid of ra and dec
radius = np.radians(1.75)

ra = np.linspace(np.pi/2.-radius, np.pi/2.+radius, 101)
dec = np.linspace(-radius, radius, 101)
ra, dec = np.meshgrid(ra,dec)
ra = np.ravel(ra)
dec = np.ravel(dec)


chipNames = myCamCoords.findChipName(ra=ra, dec=dec, epoch=epoch, camera=camera, obs_metadata=obs_metadata)

# rotate 45 degrees
obs_metadata.rotSkyPos = obs_metadata.rotSkyPos+np.pi/4.

chipNames2 = myCamCoords.findChipName(ra=ra, dec=dec, epoch=epoch, camera=camera, obs_metadata=obs_metadata)


plt.figure()
g1 = np.where(chipNames)
plt.plot(ra[g1], dec[g1],'ko')

plt.figure()
g1 = np.where(chipNames2)
plt.plot(ra[g1], dec[g1],'ko')
plt.show()
