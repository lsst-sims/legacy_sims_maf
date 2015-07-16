
__all__ = ['bundleList2Dicts']


def list2Dict(inlist):
    result = {}
    for i,oneList in enumerate(inlist):
        result[i] = oneList
    return result

def bundleList2Dicts(bundleList):
    """
    Take a list of metricBundle objects and consolidate them into
    lists with the same sqlConstriant
    """
    sqls = list(set([b.sqlconstraint for b in bundleList ]))

    resultDict = {}
    for sql in sqls:
        resultDict[sql] = []

    for bundle in bundleList:
        resultDict[bundle.sqlconstraint].append(bundle)

    for key in resultDict:
        resultDict[key] = list2Dict(resultDict[key])

    return resultDict
