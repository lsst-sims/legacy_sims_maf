from __future__ import print_function
from builtins import str
from builtins import zip
from .database import Database


__all__ = ['SimpleDatabase']


class SimpleDatabase(Database):
    def __init__(self, database, driver='sqlite', host=None, port=None, dbTables=None, 
                 defaultTable='observations', *args, **kwargs):
        if 'defaultdbTables' in kwargs:
            defaultdbTables = kwargs.get('defaultdbTables')
            # Remove this kwarg since we're sending it on explicitly
            del kwargs['defaultdbTables']
        else:
            defaultdbTables = {'observations': ['observations', 'index']}

        super(SimpleDatabase, self).__init__(driver=driver, database=database, host=host, port=port,
                                             dbTables=dbTables,
                                             defaultdbTables=defaultdbTables,
                                             *args, **kwargs)
        self.defaultTable = defaultTable

    def fetchMetricData(self, colnames, sqlconstraint, groupBy=None,
                        tableName=None, **kwargs):
        """
        Fetch 'colnames' from 'tableName'.
        """
        if tableName is None:
            tableName = self.defaultTable

        table = self.tables[tableName]
        if groupBy is not None:
            metricdata = table.query_columns_Array(chunk_size = self.chunksize,
                                                   constraint = sqlconstraint,
                                                   colnames = colnames, groupByCol = groupBy)
        else:
            metricdata = table.query_columns_Array(chunk_size = self.chunksize,
                                                   constraint = sqlconstraint,
                                                   colnames = colnames)
        return metricdata
