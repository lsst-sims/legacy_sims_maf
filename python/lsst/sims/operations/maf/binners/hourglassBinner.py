import numpy as np
import matplotlib.pyplot as plt
try:
    import astropy.io.fits as pyf
except ImportError:
    import pyfits as pyf
    
from .uniBinner import UniBinner

class HourglassBinner(UniBinner):
    """Binner to make the filter hourglass plots """

    def __init__(self, verbose=True):
        super(HourglassBinner,self).__init__(verbose=verbose)
        self.binnertype='HOUR'
        self.nbins=1
        
    def plotHour(self, metricValue, title='', xlabel='MJD (day)', ylabel='Hours from local midnight', filter2color={'u':'purple','g':'blue','r':'green','i':'cyan','z':'orange','y':'red'}):
        """expect a tuple to unpack for the metricValue from hourglassMetric  """
        f = plt.figure()
        ax = f.add_subplot(111)
        
        pernight, perfilter = metricValue
        
        plt.plot(pernight['mjd'], (pernight['twi6_rise']-pernight['midnight'])*24., 'blue' )
        plt.plot(pernight['mjd'], (pernight['twi6_set']-pernight['midnight'])*24. , 'blue' )
        
        plt.plot(pernight['mjd'], (pernight['twi12_rise']-pernight['midnight'])*24., 'yellow' )
        plt.plot(pernight['mjd'], (pernight['twi12_set']-pernight['midnight'])*24. , 'yellow' )

        plt.plot(pernight['mjd'], (pernight['twi18_rise']-pernight['midnight'])*24., 'red' )
        plt.plot(pernight['mjd'], (pernight['twi18_set']-pernight['midnight'])*24., 'red'  )

        plt.plot(pernight['mjd'], pernight['moonPer']/100.-5.5, 'black')
        y = (perfilter['mjd']-perfilter['midnight'])*24.
        for i in np.arange(0,perfilter.size,2):
            plt.plot([perfilter['mjd'][i],perfilter['mjd'][i+1] ],[y[i],y[i+1]] , filter2color[perfilter['filter'][i]] )
        
        for i,key in enumerate(['u','g','r','i','z','y']):
            plt.text(1.05,.9-i*.07, key, color=filter2color[key], transform = ax.transAxes)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        fig=plt.gcf()
        return fig.number
    
    def writeMetricData(self, outfilename, metricValues,
                        comment='', metricName='',
                        simDataName='', metadata='', 
                        int_badval=-666, badval=-666., dt=np.dtype('float64')):
        pass
    def readMetricData(self, infilename):
        pass
    
