
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
        toRemove = []
        for bundle in bundleList:
            checks = [bundleMatch(bundle,newDict[bd]) for bd in newDict]
            if False not in checks:
                toRemove.append(bundle)
                newDict[counter] = bundle
                counter += 1
        # Can't remove things until finished iterating over list
        for bundle in toRemove:
            bundleList.remove(bundle)

        result.append(newDict)

    return result
