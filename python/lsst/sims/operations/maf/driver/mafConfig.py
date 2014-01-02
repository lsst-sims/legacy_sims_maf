import lsst.pex.config as pexConfig

class MafConfig(pexConfig.Config):
    """Using pexConfig to set MAF configuration parameters"""
    dbAddress = pexConfig.Field("Address to the database to query." , str, '')
    outputDir = pexConfig.Field("Location to write MAF output", str, '')
    opsimNames = pexConfig.ListField("Which opsim runs should be analyzed", str, ['opsim_3_61'])
    
    # Manually unrolling loop since pex can't take a list of lists.
    
    grid1 = pexConfig.Field("", dtype=str, default='') # Change this to a choiceField? Or do we expect users to make new grids? 
    kwrdsForGrid1 = pexConfig.Field("", dtype=str, default='')
    metricsForGrid1 = pexConfig.ListField("", dtype=str, default=[''])
    constraintsForGrid1 = pexConfig.ListField("", dtype=str, default=[''])
    metricParamsForGrid1 = pexConfig.ListField("", dtype=str, default=[''])
    metricKwrdsForGrid1 = pexConfig.ListField("", dtype=str, default=[''])
    
    grid2 = pexConfig.Field("", dtype=str, default='')
    kwrdsForGrid2 = pexConfig.Field("", dtype=str, default='')
    metricsForGrid2 = pexConfig.ListField("", dtype=str, default=[''])
    constraintsForGrid2 = pexConfig.ListField("", dtype=str, default=[''])
    metricParamsForGrid2 = pexConfig.ListField("", dtype=str, default=[''])
    metricKwrdsForGrid2 = pexConfig.ListField("", dtype=str, default=[''])

    grid3 = pexConfig.Field("", dtype=str, default='')
    kwrdsForGrid3 = pexConfig.Field("", dtype=str, default='')
    metricsForGrid3 = pexConfig.ListField("", dtype=str, default=[''])
    constraintsForGrid3 = pexConfig.ListField("", dtype=str, default=[''])
    metricParamsForGrid3 = pexConfig.ListField("", dtype=str, default=[''])
    metricKwrdsForGrid3 = pexConfig.ListField("", dtype=str, default=[''])

    grid4 = pexConfig.Field("", dtype=str, default='')
    kwrdsForGrid4 = pexConfig.Field("", dtype=str, default='')
    metricsForGrid4 = pexConfig.ListField("", dtype=str, default=[''])
    constraintsForGrid4 = pexConfig.ListField("", dtype=str, default=[''])
    metricParamsForGrid4 = pexConfig.ListField("", dtype=str, default=[''])
    metricKwrdsForGrid4 = pexConfig.ListField("", dtype=str, default=[''])

    grid5 = pexConfig.Field("", dtype=str, default='')
    kwrdsForGrid5 = pexConfig.Field("", dtype=str, default='')
    metricsForGrid5 = pexConfig.ListField("", dtype=str, default=[''])
    constraintsForGrid5 = pexConfig.ListField("", dtype=str, default=[''])
    metricParamsForGrid5 = pexConfig.ListField("", dtype=str, default=[''])
    metricKwrdsForGrid5 = pexConfig.ListField("", dtype=str, default=[''])

    grid6 = pexConfig.Field("", dtype=str, default='')
    kwrdsForGrid6 = pexConfig.Field("", dtype=str, default='')
    metricsForGrid6 = pexConfig.ListField("", dtype=str, default=[''])
    constraintsForGrid6 = pexConfig.ListField("", dtype=str, default=[''])
    metricParamsForGrid6 = pexConfig.ListField("", dtype=str, default=[''])
    metricKwrdsForGrid6 = pexConfig.ListField("", dtype=str, default=[''])

    grid7 = pexConfig.Field("", dtype=str, default='')
    kwrdsForGrid7 = pexConfig.Field("", dtype=str, default='')
    metricsForGrid7 = pexConfig.ListField("", dtype=str, default=[''])
    constraintsForGrid7 = pexConfig.ListField("", dtype=str, default=[''])
    metricParamsForGrid7 = pexConfig.ListField("", dtype=str, default=[''])
    metricKwrdsForGrid7 = pexConfig.ListField("", dtype=str, default=[''])

    grid8 = pexConfig.Field("", dtype=str, default='')
    kwrdsForGrid8 = pexConfig.Field("", dtype=str, default='')
    metricsForGrid8 = pexConfig.ListField("", dtype=str, default=[''])
    constraintsForGrid8 = pexConfig.ListField("", dtype=str, default=[''])
    metricParamsForGrid8 = pexConfig.ListField("", dtype=str, default=[''])
    metricKwrdsForGrid8 = pexConfig.ListField("", dtype=str, default=[''])

    grid9 = pexConfig.Field("", dtype=str, default='')
    kwrdsForGrid9 = pexConfig.Field("", dtype=str, default='')
    metricsForGrid9 = pexConfig.ListField("", dtype=str, default=[''])
    constraintsForGrid9 = pexConfig.ListField("", dtype=str, default=[''])
    metricParamsForGrid9 = pexConfig.ListField("", dtype=str, default=[''])
    metricKwrdsForGrid9 = pexConfig.ListField("", dtype=str, default=[''])

    grid10 = pexConfig.Field("", dtype=str, default='')
    kwrdsForGrid10 = pexConfig.Field("", dtype=str, default='')
    metricsForGrid10 = pexConfig.ListField("", dtype=str, default=[''])
    constraintsForGrid10 = pexConfig.ListField("", dtype=str, default=[''])
    metricParamsForGrid10 = pexConfig.ListField("", dtype=str, default=[''])
    metricKwrdsForGrid10 = pexConfig.ListField("", dtype=str, default=[''])

    
    # Up to 10 possible grids?  Is that more than enough?


    
