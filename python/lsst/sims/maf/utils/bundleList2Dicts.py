
__all__ = ['bundleMatch', 'bundleList2Dicts']

def bundleMatch(bundle1, bundle2):
    """
    Check if 2 metric bundle objects are compatible to be in the same
    metricBundleGroup.
    """
    result = True
    if bundle1.sqlconstraint != bundle2.sqlconstraint:
        return False
    for stacker1 in bundle1.stackerList:
        for stacker2 in bundle2.stackerList:
            if stacker1.__class__.__name__ == stacker2.__class__.__name__:
                if stacker1 != stacker2:
                    return False

    # Possible bug in slicer comparison, with != and == giving same results
    if bundle1.slicer == bundle2.slicer:
        pass
    else:
        return False

    return result


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
