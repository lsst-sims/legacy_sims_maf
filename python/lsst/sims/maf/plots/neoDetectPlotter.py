import numpy as np
import matplotlib.pyplot as plt
from .plotHandler import BasePlotter
from matplotlib.patches import Ellipse

__all__ = ['NeoDetectPlotter']

class NeoDetectPlotter(BasePlotter):
    def __init__(self, step=.01, eclipMax=10., eclipMin=-10.):

        """
        eclipMin/Max:  only plot observations within X degrees of the ecliptic plane
        """

        self.plotType = 'neoxyPlotter'
        self.objectPlotter = True
        self.defaultPlotDict = {'title':None, 'xlabel':'X (AU)',
                                'ylabel':'Y (AU)', 'xMin':-1.5, 'xMax':1.5,
                                'yMin':-.25, 'yMax':2.5}
        self.filter2color={'u':'purple','g':'blue','r':'green',
                           'i':'cyan','z':'orange','y':'red'}
        self.filterColName = 'filter'
        self.step = step
        self.eclipMax = np.radians(eclipMax)
        self.eclipMin = np.radians(eclipMin)

    def __call__(self, metricValue, slicer,userPlotDict, fignum=None):

        fig = plt.figure(fignum)
        ax = fig.add_subplot(111)


        inPlane = np.where( (metricValue[0].data['eclipLat'] >= self.eclipMin) &
                            (metricValue[0].data['eclipLat'] <= self.eclipMax))


        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        planetProps = {'Earth': 1., 'Venus':0.72, 'Mars':1.52, 'Mercury':0.39}

        planets = []
        for prop in planetProps:
            planets.append(Ellipse((0,0), planetProps[prop]*2, planetProps[prop]*2, fill=False ))

        for planet in planets:
            ax.add_artist(planet)

        # Let's make a 2-d histogram in polar coords, then convert and display in cartisian

        rStep = self.step
        Rvec = np.arange(0,plotDict['xMax']+rStep, rStep)
        thetaStep = np.radians(3.5)
        thetavec = np.arange(0,2*np.pi+thetaStep, thetaStep)-np.pi

        # array to hold histogram values
        H = np.zeros( (thetavec.size, Rvec.size), dtype=float)

        Rgrid,thetagrid = np.meshgrid(Rvec,thetavec)

        xgrid = Rgrid*np.cos(thetagrid)
        ygrid = Rgrid*np.sin(thetagrid)


        for dist,x,y in zip(metricValue[0].data['NEOGeoDist'][inPlane],metricValue[0].data['NEOHelioX'][inPlane],
                            metricValue[0].data['NEOHelioY'][inPlane]):

            theta = np.arctan2(y-1., x)
            #theta_ind = np.searchsorted(thetavec, theta)
            #r_ind = np.searchsorted(Rvec, dist)
            diff = np.abs(thetavec - theta)
            thetaToUse = thetavec[np.where(diff == diff.min())]
            # This is a slow where-clause, should be possible to speed it up using
            # np.searchsorted+clever slicing or hist2d to build up the map.
            good = np.where( (thetagrid == thetaToUse) & (Rgrid <= dist))
            H[good] += 1


        # Set the under value to white
        myCmap = plt.cm.get_cmap('jet')
        myCmap.set_under('w')
        blah = ax.pcolormesh(xgrid,ygrid+1,H, cmap=myCmap, vmin=.001)
        cb = plt.colorbar(blah, ax=ax)

        ax.set_xlabel(plotDict['xlabel'])
        ax.set_ylabel(plotDict['ylabel'])
        ax.set_title(plotDict['title'])
        ax.set_ylim([plotDict['yMin'], plotDict['yMax']])
        ax.set_xlim([plotDict['xMin'], plotDict['xMax']])

        ax.plot([0],[1],marker='o', color='b')
        ax.plot([0],[0], marker='o', color='y')




        return fig.number
