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
        self.verbose = verbose
        # Connect to database
        # for sqlite, connecting to non-existent database creates it automatically
        if trackingDbAddress is None:
            # Default is a file in the current directory.
            dbfile = os.path.join(os.getcwd(), 'trackingDb_sqlite.db')
            self.trackingDbAddress = 'sqlite:///' + dbfile
        else:
            self.trackingDbAddress = trackingDbAddress
        engine = create_engine(self.trackingDbAddress, echo=verbose)
        if self.verbose:
            print 'Created or connected to MAF tracking database at %s' %(self.trackingDbAddress)
        self.Session = sessionmaker(bind=engine)
        self.session = self.Session()
        # Create the tables, if they don't already exist.
        try:
            Base.metadata.create_all(engine)
        except DatabaseError:
            raise DatabaseError("Cannot create a database at %s. Check directory exists." %(trackingDbAddress))

    def close(self):
        self.session.close()

    def addRun(self, opsimRun, opsimComment, mafComment, mafDir, override=False):
        """
        Add a run to the tracking database.
        """
        if opsimRun is None:
            opsimRun = 'NULL'
        if opsimComment is None:
            opsimComment = 'NULL'
        if mafComment is None:
            mafComment = 'NULL'
        # Test if mafDir already exists in database (unless directed not to check via override).
        if not override:
            prevruns = self.session.query(RunRow).filter_by(mafDir=mafDir).all()
            if len(prevruns) > 0:
                runIds = []
                for run in prevruns:
                    runIds.append(run.mafRunId)
                print 'This maf directory %s is already present in tracking db with mafRunId(s) %s.' %(mafDir, runIds)
                print 'Not currently adding this run to tracking DB (use override=True to add anyway).'
                return runIds[0]
        # Run did not exist in database or we received override: add it.
        runinfo = RunRow(opsimRun=opsimRun, opsimComment=opsimComment, mafComment=mafComment, mafDir=mafDir)
        self.session.add(runinfo)
        self.session.commit()
        return runinfo.mafRunId

    def delRun(self, runId):
        """
        Remove a run from the tracking database.
        """
        runinfo = self.session.query(RunRow).filter_by(mafRunId=runId).all()
        if len(runinfo) == 0:
            raise Exception('Could not find run with mafRunId %d' %(runId))
        if len(runinfo) > 1:
            raise Exception('Found more than one run with mafRunId %d' %(runId))
        print 'Removing run info for runId %d ' %(runId)
        print ' ', runinfo
        self.session.delete(runinfo[0])
        self.session.commit()
