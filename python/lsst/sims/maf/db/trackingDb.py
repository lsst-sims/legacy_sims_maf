import os, warnings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, backref
from sqlalchemy.exc import DatabaseError
import numpy as np

Base = declarative_base()

class RunRow(Base):
    """
    Define contents and format of run list table.

    (Table to list all available MAF results, along with their opsim run and some comment info.
    """
    __tablename__ = "runs"
    # Define columns in metric list table.
    mafRunId = Column(Integer, primary_key=True)
    opsimRun = Column(String)
    opsimComment = Column(String)
    mafComment = Column(String)
    mafDir = Column(String)
    def __repr__(self):
        return "<Run(mafRunId='%d', opsimRun='%s', opsimComment='%s', mafComment='%s', mafDir='%s'>" \
            %(self.mafRunId, self.opsimRun, self.opsimComment, self.mafComment, self.mafDir)

    
class TrackingDb(object):

    def __init__(self, trackingDbAddress=None, verbose=False):
        """
        Instantiate the results database, creating metrics, plots and summarystats tables.
        """
        # Connect to database
        # for sqlite, connecting to non-existent database creates it automatically
        if trackingDbAddress is None:
            self.trackingDbAddress = 'sqlite:///trackingDb_sqlite.db'
        else:
            self.trackingDbAddress = trackingDbAddress
        engine = create_engine(self.trackingDbAddress, echo=verbose)
        self.Session = sessionmaker(bind=engine)
        self.session = self.Session()
        # Create the tables, if they don't already exist.
        try:
            Base.metadata.create_all(engine)
        except DatabaseError:
            raise ValueError("Cannot create a database at %s. Check directory exists." %(trackingDbAddress))

    def close(self):
        self.session.close()

    def addRun(self, opsimRun, opsimComment, mafComment, mafDir):
        runinfo = RunRow(opsimRun=opsimRun, opsimComment=opsimComment, mafComment=mafComment, mafDir=mafDir)
        self.session.add(runinfo)
        self.session.commit()
        return runinfo.mafRunId

    def delRun(self, runId):
        try:
            runinfo = self.session.query(RunRow).filter_by(mafRunId=runId).one()
        except Exception as e:
            print ''
            print 'Could not find run with mafRunId %d' %(runId)
            print ''
            raise e
        print 'Removing run info: ', runinfo
        self.session.delete(runinfo)
        self.session.commit()
