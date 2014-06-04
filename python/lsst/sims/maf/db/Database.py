import warnings
from .Table import Table

def getDbAddress(connectionName='SQLITE_OPSIM', dbLoginFile=None):
    """Utility to get the dbAddress info corresponding to 'connectionName' from a dbLogin file.

    connectionName is the name given to a sqlalchemy connection string in the file
        (default 'SQLITE_OPSIM').
    dbLoginFile is the file location (default None will try to use $HOME/dbLogin). """
    # The dbLogin file is a file containing simple 'names' corresponding to sqlite connection engine
    #  strings.
    # Example:
    # SQLITE_OPSIM sqlite:///opsim.sqlite
    # MYSQL_OPSIM mysql://lsst:lsst@localhost/opsim
    #  More information on sqlalchemy connection strings can be found at
    #  http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html
    if dbLoginFile is None:
        # Try default location in home directory.
        dbLoginFile = os.path.join(os.getenv("HOME"), 'dbLogin')
    f = open(dbLoginFile, 'r')
    for l in f:
        els = l.rstrip().split()
        if els[0] == connectionName:
            dbAddress = els[1]
    return dbAddress


class Database(object):
    """Base class for database access."""
    def __init__(self, dbAddress, dbTables=None, defaultdbTables=None,
                 chunksize=1000000, **kwargs):
        """
        Instantiate database object to handle queries of the database.

        dbAddress = sqlalchemy connection string to database
        dbTables = dictionary of names of tables in the code : [names of tables in the database, primary keys]
        """
        self.dbAddress = dbAddress
        self.chunksize = chunksize
        # Add default values to provided input dictionaries (if not present in input dictionaries)
        if dbTables == None:
            self.dbTables = defaultdbTables
        else:
            self.dbTables = dbTables
            if defaultdbTables is not None:
                for k in defaultdbTables:
                    if k not in dbTables:
                        self.dbTables[k] = defaultdbTables[k]
        # Connect to database tables and store connections.
        if self.dbTables is None:
            self.tables = None
        else:
            self.tables = {}
            for k in self.dbTables:
                if len(self.dbTables[k]) != 2:
                    raise Exception('Need table name plus primary key for each value in dbTables. Missing data for %s:%s'
                                    %(k, self.dbTables[k]))
                self.tables[k] = Table(self.dbTables[k][0], self.dbTables[k][1], self.dbAddress)

    def fetchMetricData(self, colnames, sqlconstraint, **kwargs):
        """Get data from database that is destined to be used for metric evaluation.
        """
        raise NotImplementedError('Implement in subclass')

    def fetchConfig(self, *args, **kwargs):
        """Get config (metadata) info on source of data for metric calculation.
        """
        raise NotImplementedError('Implement in subclass')
                
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
