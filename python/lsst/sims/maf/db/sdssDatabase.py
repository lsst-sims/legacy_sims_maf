from .Database import Database
import numpy as np

class SdssDatabase(Database):
    """Connect to the stripe 82 database"""
    def __init__(self, dbAddress,
                 dbTables={'clue.dbo.viewStripe82JoinAll':['viewStripe82JoinAll','id']},
                 defaultdbTables=None,
                 chunksize=1000000, **kwargs):
       super(SdssDatabase,self).__init__(dbAddress,dbTables=dbTables,defaultdbTables=defaultdbTables,
                                        chunksize=chunksize,**kwargs )


    
    def fetchMetricData(self, colnames, sqlconstraint, groupBy='', **kwargs):
        """Get data for metric"""
        table = self.tables['clue.dbo.viewStripe82JoinAll']
        # MSSQL doesn't seem to like double quotes?
        sqlconstraint = sqlconstraint.replace('"', "'")
        data = table.query_columns_Array(chunk_size = self.chunksize,
                                         constraint = sqlconstraint,
                                         colnames = colnames,
                                         groupByCol = groupBy)
        # Toss columns with NaNs.
        if cleanNaNs:
            for col in colnames:
                good = np.where(np.isnan(data[col]) == False)
                data = data[good]
        return data
    
