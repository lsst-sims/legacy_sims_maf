
__all__ = ['bundleMatch', 'bundleList2Dicts']

def bundleMatch(bundle1, bundle2):
    """
    Check if 2 metric bundle objects are compatible to be in the same
    metricBundleGroup.
    """
    result = True
    if bundle1.sqlconstraint != bundle2.sqlconstraint:
        result = False
    for stacker1 in bundle1.stackerList:
        for stacker2 in bundle2.stackerList:
            if stacker1.name == stacker2.name:
                if stacker1 != stacker2:
                    result = False
    return result

def bundleList2Dicts(bundleList):
    """
    Take a list of metricBundle objects and consolidate them into compatible
    dictionaries.
    """
    result = []

    while len(bundleList) != 0:
        counter = 0
        newDict = {counter:bundleList[0]}
        counter += 1
        bundleList.remove(bundleList[0])

        for bundle in bundleList:
            compat = True
            for bd in newDict:
                if bundleMatch(bundle,newDict[bd]) is False:
                    compat = False
            if compat:
                newDict[counter] = bundle
                counter += 1
                bundleList.remove(bundle)

        result.append(newDict)

    return result
