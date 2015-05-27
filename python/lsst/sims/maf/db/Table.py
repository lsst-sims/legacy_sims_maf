__author__ = 'simon'

import numpy as np
from sqlalchemy.engine import url
from sqlalchemy import func
from sqlalchemy.sql import expression

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from lsst.sims.catalogs.generation.db import CatalogDBObject, ChunkIterator

class Table(CatalogDBObject):
    skipRegistration = True
    objid = 'sims_maf'

    def __init__(self, tableName, idColKey, dbAddress, typeOverRide=None, verbose=False):
        """
        Initialize an object for querying OpSim databases

        Keyword arguments:
        @param tableName: Name of the table to query
        @param idColKey:  Primary key for table
        @param dbAddress: A string indicating the location of the data to query.
                          This should be a database connection string.
        """
        self.idColKey = idColKey
        self.dbAddress = dbAddress
        self.tableid = tableName

        if typeOverRide is not None:
            self.dbTypeMap.update(typeOverRide)
        dbUrl = url.make_url(dbAddress)
        super(Table, self).__init__(driver=dbUrl.get_dialect().name,
                                    **dbUrl.translate_connect_args())

    def _get_column_query(self, doGroupBy, colnames=None, aggregate=func.min):
        # Build the sql query - including adding all column names, if columns were None.
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
            raise ValueError('entries in colnames must be in self.columnMap', self.columnMap)

        # Get the first query
        idColName = self.columnMap[self.idColKey]
        if idColName in vals:
            idLabel = self.idColKey
        else:
            idLabel = idColName

        #SQL server requires an aggregate on all columns if a group by clause is used.
        #Modify the columnMap to take care of this.  The default is MIN, but it shouldn't
        #matter since the entries are almost identical (except for proposalId).
        #Added double-quotes to handle column names that start with a number.
        if doGroupBy:
            query = self.session.query(aggregate(self.table.c[idColName]).label(idLabel))
        else:
            query = self.session.query(self.table.c[idColName].label(idLabel))
        for col, val in zip(colnames, vals):
            if val is idColName:
                continue
            #Check if this is a default column.
            if val == col:
                #If so, use the column in the table to take care of DB specific column
                #naming conventions (capitalization, spaces, etc.)
                if doGroupBy:
                    query = query.add_column(aggregate(self.table.c[col]).label(col))
                else:
                    query = query.add_column(self.table.c[col].label(col))
            else:
                #If not, assume that the user has specified the column correctly
                if doGroupBy:
                    query = query.add_column(aggregate(expression.literal_column(val)).label(col))
                else:
                    query = query.add_column(expression.literal_column(val).label(col))
        return query

    def query_columns_Iterator(self, colnames=None, chunk_size=None, constraint=None, groupByCol=None, numLimit=None):
        doGroupBy = not groupByCol is None
        query = self._get_column_query(doGroupBy, colnames=colnames)
        if constraint is not None:
            query = query.filter(constraint)

        if doGroupBy:
            #Either group by a column that gives unique visits
            query = query.group_by(self.table.c[groupByCol])
        if numLimit:
            query = query.limit(numLimit)
        return ChunkIterator(self, query, chunk_size)


    def query_columns_Array(self, colnames=None, chunk_size=1000000, constraint=None,
                            groupByCol=None, numLimit=None):
        """Same as query_columns, but returns a numpy rec array instead. """
        # Query the database, chunk by chunk (to reduce memory footprint).
        # If colnames == None, then will retrieve all columns in table.
        results = self.query_columns_Iterator(colnames=colnames, chunk_size=chunk_size,
                                              constraint=constraint, groupByCol=groupByCol,
                                              numLimit=numLimit)
        rescount = 0
        chunkList = []
        for result in results:
            chunkList.append(result)
            rescount += 1
        # Merge results of chunked queries.
        if rescount > 0:
            simdata = np.hstack(chunkList)
        else: # If there were no results from query, return an empty array
            dt = ['float']*len(colnames)
            simdata = np.zeros(0, dtype=zip(colnames,dt))
        return simdata
