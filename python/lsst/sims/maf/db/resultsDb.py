import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, backref

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
    runName = Column(String)
    sqlConstraint = Column(String)
    metricMetadata = Column(String)
    metricDataFile = Column(String)
    def __repr__(self):
        return "<Metric(metricId='%d', metricName='%s', slicerName='%s', runName='%s', sqlConstraint='%s', metadata='%s', metricDataFile='%s')>" % (self.metricId, self.metricName, self.slicerName, self.runName, self.sqlConstraint, self.metricMetadata, self.metricDataFile)
        
class PlotRow(Base):
    """
    Define contents and format of plot list table.

    (Table to list all plots, link them to relevant metrics in MetricList, and provide info on filename).
    """
    __tablename__ = "plots"
    # Define columns in plot list table.
    id = Column(Integer, primary_key=True)
    # Matches metricID in MetricList table.
    metricId = Column(Integer, ForeignKey('metrics.metricId'))
    plotType = Column(String)
    plotFile = Column(String)
    metric = relationship("MetricRow", backref=backref('plots', order_by=id))
    def __repr__(self):
        return "<Plot(metricId='%d', plotType='%s', plotFile='%s')>" % (self.metricId, self.plotType, self.plotFile)

    
class SummaryStatRow(Base):
    """
    Define contents and format of the summary statistics table.

    (Table to list link summary stats to relevant metrics in MetricList, and provide summary stat name, value and potentially a comment).
    """
    __tablename__ = "summarystats"
    # Define columns in plot list table.
    id = Column(Integer, primary_key=True)
    # Matches metricID in MetricList table.
    metricId = Column(Integer, ForeignKey('metrics.metricId'))
    summaryName = Column(String)
    summaryValue = Column(Float)
    metric = relationship("MetricRow", backref=backref('summarystats', order_by=id))
    def __repr__(self):
        return "<SummaryStat(metricId='%d', summaryName='%s', summaryValue='%f')>" % (self.metricId, self.summaryName, self.summaryValue)

    
class ResultsDb(object):
    def __init__(self, outDir= '.', resultsDbAddress=None, verbose=True):
        # Connect to database
        # for sqlite, connecting to non-existent database creates it automatically
        if resultsDbAddress is None:
            self.resultsDbAddress = 'sqlite:///' + os.path.join(outDir, 'resultsDb_sqlite.db')
        else:
            self.resultsDbAddress = resultsDbAddress
        engine = create_engine(self.resultsDbAddress, echo=verbose)
        self.Session = sessionmaker(bind=engine)
        self.session = self.Session()
        # Create the tables.  ## what happens if tables exist?
        Base.metadata.create_all(engine)


    def addMetric(self, metricName, slicerName, runName, sqlConstraint, metricMetadata, metricDataFile):
        metricinfo = MetricRow(metricName=metricName, slicerName=slicerName,
                             sqlConstraint=sqlConstraint, metricMetadata=metricMetadata,
                             metricDataFile=metricDataFile)
        self.session.add(metricinfo)
        self.session.commit()
        return metricinfo.metricId

    def addPlot(self, metricId, plotType, plotFile):
        plotinfo = PlotRow(metricId=metricId, plotType=plotType, plotFile=plotFile)
        self.session.add(plotinfo)
        self.session.commit()

    def addSummaryStats(self, metricId, summaryName, summaryValue):
        summarystat = SummaryStatRow(metricId=metricId, summaryName=summaryName, summaryValue=summaryValue)
        self.session.add(summarystat)
        self.session.commit()

    def getSummaryStats(self, metricId=None):
        pass

    def getMetricFiles(self, metricId=None):
        pass
            
