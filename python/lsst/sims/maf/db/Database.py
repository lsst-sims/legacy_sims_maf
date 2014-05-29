import warnings
from .table import Table

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
    def __init__(self, dbAddress, dbTables=None, dbTablesIdKey=None, defaultdbTables=None, defaultdbTablesIdKey=None,
                 chunksize=1000000, **kwargs):
        """Initiate database object to handle queries of the database.

        dbAddress = sqlalchemy connection string to database
        dbTables = dictionary of (names of tables in the code) : (names of tables in the database)
        dbTableIDKey = dictionary of (names of tables in the code) : (primary key column name)
        Note that for the dbTables and dbTableIDKey there are defaults in the init --
          you can override (specific key:value pairs only if desired) by passing a dictionary
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
        if dbTablesIdKey == None:
            dbTablesIdKey = defaultdbTablesIdKey
        else:
            if defaultdbTablesIdKey is not None:
                for k in defaultdbTablesIdKey:
                    if k not in dbTablesIdKey:
                        dbTablesIdKey[k] = defaultdbTablesIdKey[k]
        # Connect to database tables and store connections.
        self.tables = {}
        for k in self.dbTables:
            if k not in dbTablesIdKey:
                raise Exception('Trying to create db table %s but have no dbTableIdKey to match' %(k))
            self.tables[k] = Table(self.dbTables[k], dbTablesIdKey[k], dbAddress)

    def fetchMetricData(self, colnames, sqlconstraint, **kwargs):
        """Get data from database that is destined to be used for metric evaluation."""
        raise NotImplementedError('Implement in subclass')

    def fetchConfig(self, *args, **kwargs):
        """Get config (metadata) info on source of data for metric calculation."""
        raise NotImplementedError('Implement in subclass')
                
    def queryDatabase(self, table, sqlQuery):
        """Execute a general sql query (useful for arbitary joins or queries not in convenience functions).
        At present, table must be specified and all columns returned by query must be part of 'table'.
        Returns numpy recarray with results.
        """
        results = table.engine.execute(sqlQuery)
        data = table._postprocess_results(results.fetchall())
        return data
