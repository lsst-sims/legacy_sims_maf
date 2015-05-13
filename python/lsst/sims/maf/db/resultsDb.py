import os, warnings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, backref
from sqlalchemy.exc import DatabaseError
import numpy as np

Base = declarative_base()

__all__ = ['MetricRow', 'DisplayRow', 'PlotRow', 'SummaryStatRow', 'ResultsDb']

class MetricRow(Base):
    """
    Define contents and format of metric list table.

    (Table to list all metrics, their metadata, and their output data files).
    """
    __tablename__ = "metrics"
    # Define columns in metric list table.
    metricId = Column(Integer, primary_key=True)
    metricName = Column(String)
    slicerName = Column(String)
    simDataName = Column(String)
    sqlConstraint = Column(String)
    metricMetadata = Column(String)
    metricDataFile = Column(String)
    def __repr__(self):
        return "<Metric(metricId='%d', metricName='%s', slicerName='%s', simDataName='%s', sqlConstraint='%s', metadata='%s', metricDataFile='%s')>" \
          %(self.metricId, self.metricName, self.slicerName, self.simDataName,
            self.sqlConstraint, self.metricMetadata, self.metricDataFile)

class DisplayRow(Base):
    """
    Define contents and format of the displays table.

    (Table to list the display properties for each metric.)
    """
    __tablename__ = "displays"
    displayId = Column(Integer, primary_key=True)
    metricId = Column(Integer, ForeignKey('metrics.metricId'))
    # Group for displaying metric (in webpages).
    displayGroup = Column(String)
    # Subgroup for displaying metric.
    displaySubgroup = Column(String)
    # Order to display metric (within subgroup).
    displayOrder = Column(Float)
    # The figure caption.
    displayCaption = Column(String)
    metric = relationship("MetricRow", backref=backref('displays', order_by=displayId))
    def __rep__(self):
        return "<Display(displayGroup='%s', displaySubgroup='%s', displayOrder='%.1f', displayCaptio\
n='%s')>" \
            %(self.displayGroup, self.displaySubgroup, self.displayOrder, self.displayCaption)

class PlotRow(Base):
    """
    Define contents and format of plot list table.

    (Table to list all plots, link them to relevant metrics in MetricList, and provide info on filename).
    """
    __tablename__ = "plots"
    # Define columns in plot list table.
    plotId = Column(Integer, primary_key=True)
    # Matches metricID in MetricList table.
    metricId = Column(Integer, ForeignKey('metrics.metricId'))
    plotType = Column(String)
    plotFile = Column(String)
    metric = relationship("MetricRow", backref=backref('plots', order_by=plotId))
    def __repr__(self):
        return "<Plot(metricId='%d', plotType='%s', plotFile='%s')>" \
          %(self.metricId, self.plotType, self.plotFile)

class SummaryStatRow(Base):
    """
    Define contents and format of the summary statistics table.

    (Table to list and link summary stats to relevant metrics in MetricList, and provide summary stat name,
    value and potentially a comment).
    """
    __tablename__ = "summarystats"
    # Define columns in plot list table.
    statId = Column(Integer, primary_key=True)
    # Matches metricID in MetricList table.
    metricId = Column(Integer, ForeignKey('metrics.metricId'))
    summaryName = Column(String)
    summaryValue = Column(Float)
    metric = relationship("MetricRow", backref=backref('summarystats', order_by=statId))
    def __repr__(self):
        return "<SummaryStat(metricId='%d', summaryName='%s', summaryValue='%f')>" \
          %(self.metricId, self.summaryName, self.summaryValue)

class ResultsDb(object):
    def __init__(self, outDir= '.', database=None, driver='sqlite', verbose=False):
        """
        Instantiate the results database, creating metrics, plots and summarystats tables.
        """
        # Connect to database
        # for sqlite, connecting to non-existent database creates it automatically
        if database is None:
            # Check for output directory, make if needed.
            if not os.path.isdir(outDir):
                os.makedirs(outDir)
            self.database = os.path.join(outDir, 'resultsDb_sqlite.db')
            self.driver = 'sqlite'
        else:
            self.database = database
            self.driver = driver

        if self.driver == 'sqlite':
            dbAddress = url.URL(self.driver, database=self.database)
        else:
            raise NotImplementedError("The capability to connect to non-sqlite db's is in progress.")

        engine = create_engine(dbAddress, echo=verbose)
        self.Session = sessionmaker(bind=engine)
        self.session = self.Session()
        # Create the tables, if they don't already exist.
        try:
            Base.metadata.create_all(engine)
        except DatabaseError:
            raise ValueError("Cannot create a database at %s. Check directory exists." %(self.database))

    def close(self):
        self.session.close()

    def updateMetric(self, metricName, slicerName, simDataName, sqlConstraint,
                  metricMetadata, metricDataFile):
        """
        Add a row to or update a row in the metrics table.

        - metricName: the name of the metric
        - sliceName: the name of the slicer
        - simDataName: the name used to identify the simData
        - sqlConstraint: the sql constraint used to select data from the simData
        - metricMetadata: the metadata associated with the metric
        - metricDatafile: the data file the metric data is stored in

        If same metric (same metricName, slicerName, simDataName, sqlConstraint, metadata)
        already exists, it does nothing.

        Returns metricId: the Id number of this metric in the metrics table.
        """
        if simDataName is None:
            simDataName = 'NULL'
        if sqlConstraint is None:
            sqlConstraint = 'NULL'
        if metricMetadata is None:
            metricMetadata = 'NULL'
        if metricDataFile is None:
            metricDataFile = 'NULL'
        # Check if metric has already been added to database.
        prev = self.session.query(MetricRow).filter_by(metricName=metricName, slicerName=slicerName,
                                                       simDataName=simDataName, metricMetadata=metricMetadata).all()
        if len(prev) == 0:
            metricinfo = MetricRow(metricName=metricName, slicerName=slicerName, simDataName=simDataName,
                                sqlConstraint=sqlConstraint, metricMetadata=metricMetadata,
                                metricDataFile=metricDataFile)
            self.session.add(metricinfo)
            self.session.commit()
        else:
            metricinfo = prev[0]
        return metricinfo.metricId

    def updateDisplay(self, metricId, displayDict, overwrite=True):
        """
        Add a row to or update a row in the displays table.

        - metricID: the metric Id of this metric in the metrics table
        - displayDict: dictionary containing the display info

        Replaces existing row with same metricId.
        """
        # Because we want to maintain 1-1 relationship between metricId's and displayDict's:
        # First check if a display line is present with this metricID.
        displayinfo = self.session.query(DisplayRow).filter_by(metricId=metricId).all()
        if len(displayinfo) > 0:
            if overwrite:
                for d in displayinfo:
                    self.session.delete(d)
            else:
                return
        # Then go ahead and add new displayDict.
        for k in displayDict:
            if displayDict[k] is None:
                displayDict[k] = 'NULL'
        displayGroup = displayDict['group']
        displaySubgroup = displayDict['subgroup']
        displayOrder = displayDict['order']
        displayCaption = displayDict['caption']
        if displayCaption.endswith('(auto)'):
            displayCaption = displayCaption.replace('(auto)', '', 1)
        displayinfo = DisplayRow(metricId=metricId,
                                 displayGroup=displayGroup, displaySubgroup=displaySubgroup,
                                 displayOrder=displayOrder, displayCaption=displayCaption)
        self.session.add(displayinfo)
        self.session.commit()

    def updatePlot(self, metricId, plotType, plotFile):
        """
        Add a row to or update a row in the plot table.

        - metricId: the metric Id of this metric in the metrics table
        - plotType: the 'type' of this plot
        - plotFile: the filename of this plot

        Remove older rows with the same metricId, plotType and plotFile.
        """
        plotinfo = self.session.query(PlotRow).filter_by(metricId=metricId, plotType=plotType,
                                                         plotFile=plotFile).all()
        if len(plotinfo) > 0:
            for p in plotinfo:
                self.session.delete(p)
        plotinfo = PlotRow(metricId=metricId, plotType=plotType, plotFile=plotFile)
        self.session.add(plotinfo)
        self.session.commit()

    def updateSummaryStat(self, metricId, summaryName, summaryValue):
        """
        Add a row to or update a row in the summary statistic table.

        - metricId: the metric ID of this metric in the metrics table
        - summaryName: the name of this summary statistic
        - summaryValue: the value for this summary statistic

        Most summary statistics will be a simple name (string) + value (float) pair.
        For special summary statistics which must return multiple values, the base name
        can be provided as 'name', together with a np recarray as 'value', where the
        recarray also has 'name' and 'value' columns (and each name/value pair is then saved
        as a summary statistic associated with this same metricId).
        """
        # Allow for special summary statistics which return data in a np structured array with
        #   'name' and 'value' columns.  (specificially needed for TableFraction summary statistic).
        if np.size(summaryValue) > 1:
            if (('name' in summaryValue.dtype.names) and ('value' in summaryValue.dtype.names)):
                for value in summaryValue:
                    summarystat = SummaryStatRow(metricId=metricId,
                                                summaryName=summaryName+' '+value['name'],
                                                summaryValue=value['value'])
                    self.session.add(summarystat)
                    self.session.commit()
            else:
                warnings.warn('Warning! Cannot save non-conforming summary statistic.')
        # Most summary statistics will be simple floats.
        else:
            if isinstance(summaryValue, float) or isinstance(summaryValue, int):
                summarystat = SummaryStatRow(metricId=metricId, summaryName=summaryName, summaryValue=summaryValue)
                self.session.add(summarystat)
                self.session.commit()
            else:
                warnings.warn('Warning! Cannot save summary statistic that is not a simple float or int')

    def getMetricIds(self):
        """
        Return all metric Ids.
        """
        metricId = []
        for m in self.session.query(MetricRow).all():
            metricId.append(m.metricId)
        return metricId

    def getSummaryStats(self, metricId=None):
        """
        Get the summary stats for all or a single metric.
        """
        if metricId is None:
            metricId = self.getMetricIds()
        if not hasattr(metricId, '__iter__'):
            metricId = [metricId,]
        summarystats = []
        for mid in metricId:
            for m, s in self.session.query(MetricRow.metricName, MetricRow.slicerName, MetricRow.metricMetadata,
                                           SummaryStatRow).\
              filter(MetricRow.metricId == SummaryStatRow.metricId).\
              filter(MetricRow.metricId == mid).all():
              summarystats.append([m.metricName, m.slicerName, m.metricMetadata, s.summaryName, s.summaryValue])
        return summarystats

    def getMetricDataFiles(self, metricId=None):
        """
        Get the metric data filenames for all or a single metric.
        """
        if metricId is None:
            metricId = self.getMetricIds()
        if not hasattr(metricId, '__iter__'):
            metricId = [metricId,]
        dataFiles = []
        for mid in metricId:
            for m in self.session.query(MetricRow).filter(MetricRow.metricId == mid).all():
                dataFiles.append(m.metricDataFile)
        return dataFiles
