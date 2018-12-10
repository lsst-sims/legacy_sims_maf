from __future__ import print_function
import inspect
import lsst.sims.maf.metrics as metrics
try:
    import mafContrib
    mafContribPresent = True
except ImportError:
    print('sims_maf_contrib cannot be imported; skipping contrib metrics')
    mafContribPresent = False

__all__ = ['makeMetricList']

def makeMetricList(outfile):

    f = open(outfile, 'w')

    print("=================", file=f)
    print("Available metrics", file=f)
    print("=================", file=f)


    print("Core LSST MAF metrics", file=f)
    print("=====================", file=f)
    print(" ", file=f)
    for name, obj in inspect.getmembers(metrics):
        if inspect.isclass(obj):
            modname = inspect.getmodule(obj).__name__
            if modname.startswith('lsst.sims.maf.metrics'):
                link = "lsst.sims.maf.metrics.html#%s.%s" % (modname, obj.__name__)
                simpledoc = inspect.getdoc(obj).split('\n')[0]
                print("- `%s <%s>`_ \n \t %s" % (name, link, simpledoc), file=f)
    print(" ", file=f)

    if mafContribPresent:
        print("Contributed mafContrib metrics", file=f)
        print("==============================", file=f)
        print(" ", file=f)
        for name, obj in inspect.getmembers(mafContrib):
            if inspect.isclass(obj):
                modname = inspect.getmodule(obj).__name__
                if modname.startswith('mafContrib') and name.endswith('Metric'):
                    link = 'http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/%s.py' % (modname.split('.')[-1])
                    simpledoc = inspect.getdoc(obj).split('\n')[0]
                    print("- `%s <%s>`_ \n  \t %s" % (name, link, simpledoc), file=f)
        print(" ", file=f)


if __name__ == '__main__':

    makeMetricList('metricList.rst')
