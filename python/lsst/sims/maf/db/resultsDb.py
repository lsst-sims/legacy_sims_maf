import os, warnings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, backref
from sqlalchemy.exc import DatabaseError

Base = declarative_base()

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

    (Table to list link summary stats to relevant metrics in MetricList, and provide summary stat name, value and potentially a comment).
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
    def __init__(self, outDir= '.', resultsDbAddress=None, verbose=False):
        """
        Instantiate the results database, creating metrics, plots and summarystats tables.
        """
        # Connect to database
        # for sqlite, connecting to non-existent database creates it automatically
        if resultsDbAddress is None:
            # Check for output directory, make if needed.
            if not os.path.isdir(outDir):
                os.makedirs(outDir)
            self.resultsDbAddress = 'sqlite:///' + os.path.join(outDir, 'resultsDb_sqlite.db')
        else:
            self.resultsDbAddress = resultsDbAddress
        engine = create_engine(self.resultsDbAddress, echo=verbose)
        self.Session = sessionmaker(bind=engine)
        self.session = self.Session()
        # Create the tables, if they don't already exist.
        try:
            Base.metadata.create_all(engine)
        except DatabaseError:
            raise ValueError("Cannot create a database at %s. Check directory exists." %(resultsDbAddress))

    def close(self):
        self.session.close()
        
    def addMetric(self, metricName, slicerName, simDataName, sqlConstraint,
                  metricMetadata, metricDataFile):
        """
        Add a row to the metrics table.
        """
        ## TODO: check if row already exists in table, and if so, don't add it again.
        metricinfo = MetricRow(metricName=metricName, slicerName=slicerName, simDataName=simDataName,
                                sqlConstraint=sqlConstraint, metricMetadata=metricMetadata,
                                metricDataFile=metricDataFile)
        self.session.add(metricinfo)
        self.session.commit()
        return metricinfo.metricId

    def addPlot(self, metricId, plotType, plotFile):
        """
        Add a row to the plot table.
        """
        ## TODO: check if row already exists in table, and if so, don't add it again.
        plotinfo = PlotRow(metricId=metricId, plotType=plotType, plotFile=plotFile)
        self.session.add(plotinfo)
        self.session.commit()

    def addSummaryStat(self, metricId, summaryName, summaryValue):
        """
        Add a row to the summary statistic table.
        """
        ## TODO: check if row already exists in table, and if so, don't add it again.
        if not ((isinstance(summaryValue, float)) or isinstance(summaryValue, int)):
            warnings.warn('Cannot save non-float/non-int values for summary statistics.')
            return
        summarystat = SummaryStatRow(metricId=metricId, summaryName=summaryName, summaryValue=summaryValue)
        self.session.add(summarystat)
        self.session.commit()

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
        for mid in metricId:
            for m, s in self.session.query(MetricRow, SummaryStatRow).\
              filter(MetricRow.metricId == SummaryStatRow.metricId).\
              filter(MetricRow.metricId == mid).all():
                print 'Metric:', m.metricName, m.slicerName, m.metricMetadata                
                print ' ', s.summaryName, s.summaryValue
                
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
                print 'Metric:', m.metricName, m.slicerName, m.metricMetadata, m.metricDataFile
                dataFiles.append(m.metricDataFile)
        return dataFiles
