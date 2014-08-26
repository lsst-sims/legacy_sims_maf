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
        sqlQuery = 'select '+','.join(colnames)+' from '+ self.tables.keys()[0]+' where '+sqlconstraint
        if groupBy != '':
            sqlQuery = sqlQuery + 'group by '+groupBy
        sqlQuery = sqlQuery+' ;'
        # MSSQL doesn't seem to like double quotes?
        sqlQuery=sqlQuery.replace('"', "'")
        data = self.queryDatabase(self.tables.keys()[0] , sqlQuery)
        for col in colnames:
            good = np.where(np.isnan(data[col]) == False)
            data = data[good]
        return data
    
