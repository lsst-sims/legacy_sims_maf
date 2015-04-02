import numpy as np
import matplotlib.pyplot as plt
import warnings

from .uniSlicer import UniSlicer

__all__ = ['HourglassSlicer']

class HourglassSlicer(UniSlicer):
    """Slicer to make the filter hourglass plots """

    def __init__(self, verbose=True, badval=-666):
        # Inherits from UniSlicer, so nslice=1 and only one 'slice'.
        super(HourglassSlicer,self).__init__(verbose=verbose, badval=badval)
        self.columnsNeeded=[]
        self.slicerName='HourglassSlicer'

    def plotData(self, metricValues, figformat='png', filename=None, savefig=True, dpi=600, thumbnail=True, **kwargs):
        """
        Override base plotData to generate hourglass plot.
        """
        filenames=[]
        filetypes=[]
        figs={}
        if not isinstance(metricValues[0], dict):
            warnings.warn('HourglassSlicer did not get dict to plot, returning False')
            return {'figs':figs, 'filenames':filenames, 'filetypes':filetypes}
        figs['hourglass'] = self.plotHour(metricValues, **kwargs)
        if savefig:
            outfile = filename+'_hr'+'.'+figformat
            plt.savefig(outfile, figformat=figformat, dpi=dpi)
            if thumbnail:
                thumbfile = outfile.split('/')
                thumbfile[-1] = 'thumb.'+thumbfile[-1][:-4]
                thumbfile = '/'.join(thumbfile)
                plt.savefig(thumbfile+'.'+'png', dpi=72)
            filenames.append(outfile)
            filetypes.append('hourglassPlot')
        return {'figs':figs,'filenames':filenames,'filetypes':filetypes}

    def plotHour(self, metricValue, title='', xlabel=None, ylabel='Hours from local midnight',
                 filter2color={'u':'purple','g':'blue','r':'green','i':'cyan','z':'orange','y':'red'},
                **kwargs):
        """
        Generate the 'hourglass' plot.

        Expect a tuple 'metricValue' to unpack for the metricValue from hourglassMetric.
        """
        xlabel = 'Night - min(Night)' # Currently not able to override.
        f = plt.figure()
        ax = f.add_subplot(111)

        pernight =  metricValue[0]['pernight']
        perfilter = metricValue[0]['perfilter']

        y = (perfilter['mjd']-perfilter['midnight'])*24.
        dmin = np.floor(np.min(perfilter['mjd']))
        for i in np.arange(0,perfilter.size,2):
            plt.plot([perfilter['mjd'][i]-dmin,perfilter['mjd'][i+1]-dmin ],[y[i],y[i+1]],
                     filter2color[perfilter['filter'][i]] )
        for i,key in enumerate(['u','g','r','i','z','y']):
            plt.text(1.05,.9-i*.07, key, color=filter2color[key], transform = ax.transAxes)
        plt.plot(pernight['mjd']-dmin, (pernight['twi6_rise']-pernight['midnight'])*24.,
                 'blue', label=r'6$^\circ$ twilight' )
        plt.plot(pernight['mjd']-dmin, (pernight['twi6_set']-pernight['midnight'])*24. ,
                 'blue' )
        plt.plot(pernight['mjd']-dmin, (pernight['twi12_rise']-pernight['midnight'])*24.,
                 'yellow', label=r'12$^\circ$ twilight' )
        plt.plot(pernight['mjd']-dmin, (pernight['twi12_set']-pernight['midnight'])*24. ,
                  'yellow' )
        plt.plot(pernight['mjd']-dmin, (pernight['twi18_rise']-pernight['midnight'])*24.,
                 'red', label=r'18$^\circ$ twilight' )
        plt.plot(pernight['mjd']-dmin, (pernight['twi18_set']-pernight['midnight'])*24.,
                 'red'  )
        plt.plot(pernight['mjd']-dmin, pernight['moonPer']/100.-7, 'black', label='Moon')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        fig=plt.gcf()
        return fig.number

    def writeData(self, outfilename, metricValues, metricName='', **kwargs):
        """
        Override base write method: we don't want to save hourglass metric data.

        The data volume is too large.
        """
        pass

    def readMetricData(self, infilename):
        """
        Override base read method to 'pass': we don't save or read hourglass metric data.

        The data volume is too large.
        """
        pass

