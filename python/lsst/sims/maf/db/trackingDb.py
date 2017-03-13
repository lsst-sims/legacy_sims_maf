from __future__ import print_function
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

    (Table to list all available MAF results, along with their opsim run and some comment info.
    """
    __tablename__ = "runs"
    # Define columns in metric list table.
    mafRunId = Column(Integer, primary_key=True)
    opsimRun = Column(String)
    opsimComment = Column(String)
    mafComment = Column(String)
    mafDir = Column(String)
    opsimDate = Column(String)
    mafDate = Column(String)
    dbFile = Column(String)
    def __repr__(self):
        rstr = "<Run(mafRunId='%d', opsimRun='%s', opsimComment='%s', mafComment='%s', " % (self.mafRunId, self.opsimRun,
                                                                                          self.opsimComment, self.mafComment)
        rstr += "mafDir='%s', opsimDate='%s', mafDate='%s', dbFile='%s'>" %(self.mafDir, self.opsimDate, self.mafDate, self.dbFile)
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

    def addRun(self, opsimRun, opsimComment, mafComment, mafDir, opsimDate, mafDate, dbFile, override=False):
        """
        Add a run to the tracking database.
        """
        if opsimRun is None:
            opsimRun = 'NULL'
        if opsimComment is None:
            opsimComment = 'NULL'
        if mafComment is None:
            mafComment = 'NULL'
        if opsimDate is None:
            opsimDate = 'NULL'
        if mafDate is None:
            mafDate = 'NULL'
        if dbFile is None:
            dbFile = 'NULL'
        # Test if mafDir already exists in database (unless directed not to check via override).
        if not override:
            prevruns = self.session.query(RunRow).filter_by(mafDir=mafDir).all()
            if len(prevruns) > 0:
                runIds = []
                for run in prevruns:
                    runIds.append(run.mafRunId)
                print('This maf directory %s is already present in tracking db with mafRunId(s) %s.' %(mafDir, runIds))
                print('Not currently adding this run to tracking DB (use override=True to add anyway).')
                return runIds[0]
        # Run did not exist in database or we received override: add it.
        runinfo = RunRow(opsimRun=opsimRun, opsimComment=opsimComment, mafComment=mafComment,
                         mafDir=mafDir, opsimDate=opsimDate, mafDate=mafDate, dbFile=dbFile)
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
