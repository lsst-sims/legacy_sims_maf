import inspect
import lsst.sims.maf.metrics as metrics
try:
    import mafContrib
    mafContribPresent = True
except ImportError:
    mafContribPresent = False

__all__ = ['makeMetricList']

def makeMetricList(outfile):

    f = open(outfile, 'w')

    print >>f,  "================="
    print >>f,  "Available metrics"
    print >>f, "================="


    print >>f, "Core LSST MAF metrics"
    print >>f, "====================="
    print >>f, " "
    for name, obj in inspect.getmembers(metrics):
        if inspect.isclass(obj):
            modname = inspect.getmodule(obj).__name__
            if modname.startswith('lsst.sims.maf.metrics'):
                link = "lsst.sims.maf.metrics.html#%s.%s" % (modname, obj.__name__)
                print >>f, "- `%s <%s>`_" % (name, link)
    print >>f, " "

    if mafContribPresent:
        print >>f, "Contributed mafContrib metrics"
        print >>f, "=============================="
        print >>f, " "
        for name, obj in inspect.getmembers(mafContrib):
            if inspect.isclass(obj):
                modname = inspect.getmodule(obj).__name__
                if modname.startswith('mafContrib') and name.endswith('Metric'):
                    link = 'http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/%s.py' % (modname.split('.')[-1])
                    print >>f, "- `%s <%s>`_" % (name, link)
        print >>f, " "


if __name__ == '__main__':

    makeMetricList('metricList.rst')
