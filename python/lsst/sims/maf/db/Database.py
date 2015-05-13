import os, warnings
import inspect
from sqlalchemy.engine import url

from .Table import Table

__all__ = ['DatabaseRegistry', 'Database']

class DatabaseRegistry(type):
    """
    Meta class for databases, to build a registry of database classes.
    """
    def __init__(cls, name, bases, dict):
        super(DatabaseRegistry, cls).__init__(name, bases, dict)
        if not hasattr(cls, 'registry'):
            cls.registry = {}
        modname = inspect.getmodule(cls).__name__ + '.'
        if modname.startswith('lsst.sims.maf.db'):
            modname = ''
        else:
            if len(modname.split('.')) > 1:
                modname = '.'.join(modname.split('.')[:-1]) + '.'
            else:
                modname = modname + '.'
        databasename = modname + name
        if databasename in cls.registry:
            raise Exception('Redefining databases %s! (there are >1 database classes with the same name)'
                            %(databasename))
        if databasename not in ['BaseDatabase']:
            cls.registry[databasename] = cls
    def getClass(cls, databasename):
        return cls.registry[databasename]
    def list(cls, doc=False):
        for databasename in sorted(cls.registry):
            if not doc:
                print databasename
            if doc:
                print '---- ', databasename, ' ----'
                print inspect.getdoc(cls.registry[databasename])


class Database(object):
    """Base class for database access."""

    __metaclass__ = DatabaseRegistry

    def __init__(self, database, driver='sqlite', host=None, port=None,
                 dbTables=None, defaultdbTables=None,
                 chunksize=1000000, longstrings=False, verbose=False):
        """
        Instantiate database object to handle queries of the database.

        database = name of the database to connect to (in the case of sqlite, the filename)
        driver =  Name of database dialect+driver for sqlalchemy (e.g. 'sqlite', 'pymssql+mssql')
        host = Name of database host (optional)
        port = String port number (optional)
        dbTables = dictionary of names of tables in the code : [names of tables in the database, primary keys]
        """
        if longstrings:
            typeOverRide = {'VARCHAR':(str, 1024), 'NVARCHAR':(str, 1024),
                            'TEXT':(str, 1024), 'CLOB':(str, 1024),
                            'STRING':(str, 1024)}

        self.driver = driver
        self.host = host
        self.port = port
        self.database = database
        self.chunksize = chunksize

        # Add default values to provided input dictionaries (if not present in input dictionaries)
        if dbTables == None:
            self.dbTables = defaultdbTables
        else:
            self.dbTables = dbTables
            if defaultdbTables is not None:
                # Add defaultdbTables into dbTables
                defaultdbTables.update(self.dbTables)
                self.dbTables = defaultdbTables
        # Connect to database tables and store connections.
        # Test file exists if connecting to sqlite db.
        if self.driver.startswith('sqlite'):
            filename = self.database.replace('sqlite:','').replace('///','')
            if not os.path.isfile(filename):
                raise IOError('Sqlite database file "%s" not found.' %(filename))

        # Build sqlalchemy dbAddress string here for now.
        if self.driver == 'sqlite':
            self.dbAddress = url.URL(self.driver, database=self.database)
        else:
            raise NotImplementedError("The capability to connect to non-sqlite db's is in progress.")

        if self.dbTables is None:
            self.tables = None
        else:
            self.tables = {}
            for k in self.dbTables:
                if len(self.dbTables[k]) != 2:
                    raise Exception('Need table name plus primary key for each value in dbTables. Missing data for %s:%s'
                                    %(k, self.dbTables[k]))
                if longstrings:
                    self.tables[k] = Table(self.dbTables[k][0], self.dbTables[k][1],
                                           self.dbAddress, typeOverRide=typeOverRide,
                                           verbose=verbose)
                else:
                    self.tables[k] = Table(self.dbTables[k][0], self.dbTables[k][1],
                                           self.dbAddress, verbose=verbose)

    def fetchMetricData(self, colnames, sqlconstraint, **kwargs):
        """
        Get data from database that is destined to be used for metric evaluation.
        """
        raise NotImplementedError('Implement in subclass')

    def fetchConfig(self, *args, **kwargs):
        """
        Get config (metadata) info on source of data for metric calculation.
        """
        # Demo API (for interface with driver).
        configSummary = {}
        configDetails = {}
        return configSummary, configDetails

    def queryDatabase(self, tableName, sqlQuery):
        """
        Execute a general sql query (useful for arbitary joins or queries not in convenience functions).
        At present, 'table' (name) must be specified and all columns returned by query must be part of 'table'.
        Returns numpy recarray with results.
        """
        t = self.tables[tableName]
        results = t.engine.execute(sqlQuery)
        data = t._postprocess_results(results.fetchall())
        return data
