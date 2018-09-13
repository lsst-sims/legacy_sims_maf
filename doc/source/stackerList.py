from __future__ import print_function
import inspect
import lsst.sims.maf.stackers as stackers
try:
    import mafContrib
    mafContribPresent = True
except ImportError:
    mafContribPresent = False

__all__ = ['makeStackerList']

def makeStackerList(outfile):

    f = open(outfile, 'w')

    print("==================", file=f)
    print("Available stackers", file=f)
    print("==================", file=f)


    print("Core LSST MAF stackers", file=f)
    print("======================", file=f)
    print(" ", file=f)
    for name, obj in inspect.getmembers(stackers):
        if inspect.isclass(obj):
            modname = inspect.getmodule(obj).__name__
            if modname.startswith('lsst.sims.maf.stackers'):
                if name == 'ColInfo':
                    continue
                if name == "StackerRegistry":
                    continue
                link = "lsst.sims.maf.stackers.html#%s.%s" % (modname, obj.__name__)
                simpledoc = inspect.getdoc(obj).split('\n')[0]
                print("- `%s <%s>`_ \n \t %s" % (name, link, simpledoc), file=f)
                try:
                    print('\n\t Adds columns: %s' % (obj.colsAdded), file=f)
                except AttributeError:
                    pass
    print(" ", file=f)

    if mafContribPresent:
        print("Contributed mafContrib stackers", file=f)
        print("===============================", file=f)
        print(" ", file=f)
        for name, obj in inspect.getmembers(mafContrib):
            if inspect.isclass(obj):
                modname = inspect.getmodule(obj).__name__
                if modname.startswith('mafContrib') and name.endswith('Stacker'):
                    link = 'http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/%s.py' % (modname.split('.')[-1])
                    simpledoc = inspect.getdoc(obj).split('\n')[0]
                    print("- `%s <%s>`_ \n  \t %s" % (name, link, simpledoc), file=f)
                    try:
                        print('\n\t Adds columns: %s' % (obj.colsAdded), file=f)
                    except AttributeError:
                        pass
        print(" ", file=f)


if __name__ == '__main__':

    makeStackerList('stackerList.rst')
