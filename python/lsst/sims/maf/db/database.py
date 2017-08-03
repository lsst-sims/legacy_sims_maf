from __future__ import print_function
from future.utils import with_metaclass
import os
import inspect
import numpy as np
from sqlalchemy import func, text
from sqlalchemy import Table
from sqlalchemy.engine import reflection
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from lsst.sims.catalogs.db import DBObject


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
    def help(cls, doc=False):
        for databasename in sorted(cls.registry):
            if not doc:
                print(databasename)
            if doc:
                print('---- ', databasename, ' ----')
                print(inspect.getdoc(cls.registry[databasename]))


class Database(with_metaclass(DatabaseRegistry, DBObject)):
    """Base class for database access."""

    def __init__(self, database, driver='sqlite', host=None, port=None,
                 chunksize=1000000, longstrings=False, verbose=False):
        """
        Instantiate database object to handle queries of the database.

        database = Name of database or sqlite filename
        driver =  Name of database dialect+driver for sqlalchemy (e.g. 'sqlite', 'pymssql+mssql')
        host = Name of database host (optional)
        port = String port number (optional)        
        """
        # If it's a sqlite file, check that the filename exists.
        # This gives a more understandable error message than trying to connect to non-existent file later.
        if driver=='sqlite':
            if not os.path.isfile(database):
                raise IOError('Sqlite database file "%s" not found.' %(database))

        # Connect to database using DBObject init.
        super(Database, self).__init__(database=database, driver=driver,
                                       host=host, port=port, verbose=verbose, connection=None)

        self.dbTypeMap = {'BIGINT': (int,), 'BOOLEAN': (bool,), 'FLOAT': (float,), 'INTEGER': (int,),
                         'NUMERIC': (float,), 'SMALLINT': (int,), 'TINYINT': (int,), 'VARCHAR': (np.str, 256),
                         'TEXT': (np.str, 256), 'CLOB': (np.str, 256), 'NVARCHAR': (np.str, 256),
                         'NCLOB': (np.str, 256), 'NTEXT': (np.str, 256), 'CHAR': (np.str, 1), 'INT': (int,),
                         'REAL': (float,), 'DOUBLE': (float,), 'STRING': (np.str, 256), 'DOUBLE_PRECISION': (float,),
                         'DECIMAL': (float,)}
        if longstrings:
            typeOverRide = {'VARCHAR':(np.str, 1024), 'NVARCHAR':(np.str, 1024),
                            'TEXT':(np.str, 1024), 'CLOB':(np.str, 1024),
                            'STRING':(np.str, 1024)}
            self.dbTypeMap.update(typeOverRide)

        # Get a dict (keyed by the table names) of all the columns in each table.
        self.tableNames = reflection.Inspector.from_engine(self.connection.engine).get_table_names()
        self.tableNames += reflection.Inspector.from_engine(self.connection.engine).get_view_names()
        self.columnNames = {}
        for t in self.tableNames:
            cols = reflection.Inspector.from_engine(self.connection.engine).get_columns(t)
            self.columnNames[t] = [xxx['name'] for xxx in cols]
        # Create all the sqlalchemy table objects. This lets us see the schema and query it with types.
        self.tables = {}
        for tablename in self.tableNames:
            self.tables[tablename] = Table(tablename, self.connection.metadata, autoload=True)

    def fetchMetricData(self, colnames, sqlconstraint, table=None, **kwargs):
        """
        Get data from table that is destined to be used for metric evaluation.
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

    def query_arbitrary(self, sqlQuery, dtype=None):
        """
        Simple wrapper around execute_arbitrary for backwards compatibility.
    
        Parameters
        -----------
        sqlQuery : str
            SQL query. 
        dtype: opt, numpy dtype.
            Numpy recarray dtype. If None, then an attempt to determine the dtype will be made.
            This attempt will fail if there are commas in the data you query. 

        Returns
        -------
        numpy.recarray        
        """
        return self.execute_arbitrary(sqlQuery, dtype=dtype)

    def query_columns(self, tablename, colnames=None, sqlconstraint=None,
                            groupBy=None, aggregate=func.min, numLimit=None):
        # Build the sqlalchemy query from a single table, with various columns/constraints/etc.
        # Does NOT use a mapping between column names and database names - assumes the database names
        # are what the user will specify.
        if tablename not in self.tables:
            raise ValueError('Tablename %s not in list of available tables (%s).'
                             % (tablename, self.tables.keys()))
        if colnames is None:
            colnames = self.columnNames[tablename]
        else:
            for col in colnames:
                if col not in self.columnNames[tablename]:
                    raise ValueError("Requested column %s not available in table %s" % (col, tablename))
            if groupBy is not None:
                if groupBy not in self.columnNames[tablename]:
                    raise ValueError("GroupBy column %s is not available in table %s" % (groupBy, tablename))

        for col in colnames:
            if col == colnames[0]:
                if groupBy:
                    query = self.connection.session.query(aggregate(col))
                else:
                    query = self.connection.session.query(col)
            else:
                if groupBy:
                    query = query.add_column(aggregate(col))
                else:
                    query = query.add_column(col)

        query = query.select_from(self.tables[tablename])

        if sqlconstraint is not None:
            query = query.filter(text(sqlconstraint))

        if groupBy is not None:
            query = query.group_by(self.table.c[groupByCol])

        if numLimit is not None:
            query = query.limit(numLimit)

        # Execute query and get results.
        results = self.connection.session.execute(query).fetchall()

        # Translate results into numpy recarray.
        dtype = []
        for col in colnames:
            dt = self.dbTypeMap[self.tables[tablename].c[col].type.__visit_name__]
            dtype.append((col, ) + dt)

        return(np.rec.fromrecords(results, dtype=dtype))


