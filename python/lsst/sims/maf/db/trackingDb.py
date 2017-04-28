from __future__ import print_function
from builtins import str
from builtins import object
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.engine import url
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import DatabaseError
from lsst.daf.persistence import DbAuth

Base = declarative_base()

__all__ = ['TrackingDb']


class RunRow(Base):
    """
    Define contents and format of run list table.

    Table to list all available MAF results, along with their opsim run and some comment info.
    """
    __tablename__ = "runs"
    # Define columns in metric list table.
    mafRunId = Column(Integer, primary_key=True)
    opsimGroup = Column(String)
    opsimRun = Column(String)
    opsimComment = Column(String)
    opsimVersion = Column(String)
    opsimDate = Column(String)
    dbFile = Column(String)
    mafComment = Column(String)
    mafVersion = Column(String)
    mafDate = Column(String)
    mafDir = Column(String)
    def __repr__(self):
        rstr = "<Run(mafRunId='%d', opsimGroup='%s', opsimRun='%s', opsimComment='%s', " \
               "opsimVersion='%s', opsimDate='%s', mafComment='%s', " \
               "mafVersion='%s', mafDate='%s', mafDir='%s', dbFile='%s'>" \
                % (self.mafRunId, self.opsimGroup, self.opsimRun, self.opsimComment,
                   self.opsimVersion, self.opsimDate, self.mafComment, self.mafVersion, self.mafDate,
                   self.mafDir, self.dbFile)
        return rstr


class TrackingDb(object):

    def __init__(self, database=None, driver='sqlite', host=None, port=None,
                 trackingDbverbose=False):
        """
        Instantiate the results database, creating metrics, plots and summarystats tables.
        """
        self.verbose = trackingDbverbose
        # Connect to database
        # for sqlite, connecting to non-existent database creates it automatically
        if database is None:
            # Default is a file in the current directory.
            self.database = os.path.join(os.getcwd(), 'trackingDb_sqlite.db')
            self.driver = 'sqlite'
        else:
            self.database  = database
            self.driver = driver
            self.host = host
            self.port = port

        if self.driver == 'sqlite':
            dbAddress = url.URL(drivername=self.driver, database=self.database)
        else:
            dbAddress = url.URL(self.driver,
                            username=DbAuth.username(self.host, str(self.port)),
                            password=DbAuth.password(self.host, str(self.port)),
                            host=self.host,
                            port=self.port,
                            database=self.database)

        engine = create_engine(dbAddress, echo=self.verbose)
        if self.verbose:
            print('Created or connected to MAF tracking %s database at %s' %(self.driver, self.database))
        self.Session = sessionmaker(bind=engine)
        self.session = self.Session()
        # Create the tables, if they don't already exist.
        try:
            Base.metadata.create_all(engine)
        except DatabaseError:
            raise DatabaseError("Cannot create a %s database at %s. Check directory exists." %(self.driver, self.database))

    def close(self):
        self.session.close()

    def addRun(self, opsimGroup=None, opsimRun=None, opsimComment=None, opsimVersion=None, opsimDate=None,
               mafComment=None, mafVersion=None, mafDate=None, mafDir=None, dbFile=None):
        """Add a run to the tracking database.
        
        Parameters
        ----------
        opsimGroup : str, opt
            Set a name to group this run with (eg. "Tier 1, 2016").
        opsimRun : str, opt
            Set a name for the opsim run.
        opsimComment : str, opt
            Set a comment describing the opsim run.
        opsimVersion : str, opt
            Set the version of opsim.
        opsimDate : str, opt
            Set the date the opsim run was created.
        mafComment : str, opt
            Set a comment to describe the MAF analysis.
        mafVersion : str, opt
            Set the version of MAF used for analysis.
        mafDate : str, opt
            Set the date the MAF analysis was run.
        mafDir : str, opt
            The relative path to the MAF directory.
        dbFile : str, opt
            The relative path to the Opsim SQLite database file.

        Returns
        -------
        int
            The mafRunID stored in the database.
        """
        if opsimGroup is None:
            opsimGroup = 'NULL'
        if opsimRun is None:
            opsimRun = 'NULL'
        if opsimComment is None:
            opsimComment = 'NULL'
        if opsimVersion is None:
            opsimVersion = 'NULL'
        if opsimDate is None:
            opsimDate = 'NULL'
        if mafComment is None:
            mafComment = 'NULL'
        if mafVersion is None:
            mafVersion = 'NULL'
        if mafDate is None:
            mafDate = 'NULL'
        if mafDir is None:
            mafDir = 'NULL'
        if dbFile is None:
            dbFile = 'NULL'
        # Test if mafDir already exists in database.
        prevrun = self.session.query(RunRow).filter_by(mafDir=mafDir).all()
        if len(prevrun) > 0:
            runIds = []
            for run in prevrun:
                runIds.append(run.mafRunId)
            print('Updating information in tracking database - %s already present with runId %s.'
                  % (mafDir, runIds))
            for run in prevrun:
                self.session.delete(run)
            self.session.commit()
            runinfo = RunRow(mafRunId=runIds[0], opsimGroup=opsimGroup, opsimRun=opsimRun,
                             opsimComment=opsimComment, opsimVersion=opsimVersion, opsimDate=opsimDate,
                             mafComment=mafComment, mafVersion=mafVersion, mafDate=mafDate,
                             mafDir=mafDir, dbFile=dbFile)
        else:
            runinfo = RunRow(opsimGroup=opsimGroup, opsimRun=opsimRun,
                             opsimComment=opsimComment, opsimVersion=opsimVersion, opsimDate=opsimDate,
                             mafComment=mafComment, mafVersion=mafVersion, mafDate=mafDate,
                             mafDir=mafDir, dbFile=dbFile)
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
        print('Removing run info for runId %d ' %(runId))
        print(' ', runinfo)
        self.session.delete(runinfo[0])
        self.session.commit()
