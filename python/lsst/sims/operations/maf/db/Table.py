__author__ = 'simon'

from lsst.sims.catalogs.generation.db import DBObject, ChunkIterator
import numpy
from sqlalchemy import func
from sqlalchemy.sql import expression

class Table(DBObject):
    skipRegistration = True

    def __init__(self, tableName, idColKey, dbAddress):
        """
        Initialize an object for querying OpSim databases

        Keyword arguments:
        @param runtable: Name of the table to query
        @param dbAddress: A string indicating the location of the data to query.
                          This should be a database connection string.
        """

        if dbAddress is None:
            dbAddress = self.getDbAddress()

	self.idColKey = idColKey
        self.dbAddress = dbAddress
        self.tableid = tableName
        self._connect_to_engine()
        self._get_table()

        if self.generateDefaultColumnMap:
            self._make_default_columns()

        self._make_column_map()
        self._make_type_map()

    def _get_column_query(self, colnames=None):
        if colnames is None:
            colnames = [k for k in self.columnMap.keys()]
        try:
            vals = [self.columnMap[k] for k in colnames]
        except KeyError:
            for c in colnames:
                if c in self.columnMap.keys():
                    continue
                else:
                    print("%s not in columnMap"%(c))
            raise ValueError('entries in colnames must be in self.columnMap')

        # Get the first query
        idColName = self.columnMap[self.idColKey]
        if idColName in vals:
            idLabel = self.idColKey
        else:
            idLabel = idColName

        #SQL server requires an aggregate on all columns if a group by clause is used.
        #Modify the columnMap to take care of this.  I'm choosing MIN, but it shouldn't
        #matter since the entries are almost identical (except for proposalId???)
        #Added double-quotes to handle column names that start with a number.
        query = self.session.query(func.min(self.table.c[idColName]).label(idLabel))
        for col, val in zip(colnames, vals):
            if val is idColName:
                continue
            #Check if this is a default column.
            if val == col:
                #If so, use the column in the table to take care of DB specific column
                #naming conventions (capitalization, spaces, etc.)
                query = query.add_column(func.min(self.table.c[col]).label(col))
            else:
                #If not, assume that the user has specified the column correctly
                query = query.add_column(func.min(expression.literal_column(val)).label(col))
        return query

    def query_columns(self, colnames=None, chunk_size=None, constraint=None, groupByCol=None, numLimit=None):

        query = self._get_column_query(colnames)
        if constraint is not None:
            query = query.filter(constraint)

        if groupByCol is not None:
            #Either group by a column that gives unique visits
            query = query.group_by(groupByCol)
        #else:
            #or group by the unique id for the table.
        #    query = query.group_by(self.idColKey)
        if numLimit:
            query = query.limit(numLimit)
        return ChunkIterator(self, query, chunk_size)
