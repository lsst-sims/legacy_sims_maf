============================
Metrics in run_all.py script
============================
The `run_all.py` script included in MAF runs a very large number of useful
metrics covering metedata about observing history, survey performance, and
those from the LSST SRD. Here we will list and summarize the metics included
in this script.



SRD metrics
===========

`fOBatch <lsst.sims.maf.batches.html#lsst.sims.maf.batches.srdBatch.fOBatch>`_
--------------------------------------------------------------------------------

Metrics included:

 - WFD only, and all proposals
 - `fOArea <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.fOArea>`_
 - `fONv <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.fONv>`_

`astrometryBatch <lsst.sims.maf.batches.html#lsst.sims.maf.batches.srdBatch.astrometryBatch>`_
------------------------------------------------------------------------------------------------
Metrics included:

 -  WFD only, and all proposals
 - `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
 - `ProperMotionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ProperMotionMetric>`_
 - `ParallaxMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxMetric>`_
 - `ParallaxCoverageMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxCoverageMetric>`_
 - `ParallaxDcrDegenMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxDcrDegenMetric>`_


`rapidRevisitBatch <lsst.sims.maf.batches.html#lsst.sims.maf.batches.srdBatch.rapidRevisitBatch>`_
---------------------------------------------------------------------------------------------------
Metrics included:

  -  WFD only, and all proposals
  -  `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
  - `RapidRevisitMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.RapidRevisitMetric>`_


Cadence metrics (pairs and time)
================================

`intraNightBatch <lsst.sims.maf.batches.html#lsst.sims.maf.batches.timeBatch.intraNight>`_
--------------------------------------------------------------------------------------------
Metrics included:

 -  All proposals, and WFD+NES Only
 -  `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
 - `PairFractionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.visitGroupsMetric.PairFractionMetric>`_
 - `NRevisitsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.NRevisitsMetric>`_
 - `Median IntraNightGapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.IntraNightGapsMetric>`_
 - `NVisitsPerNightMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.tgaps.NVisitsPerNightMetric>`_
 - `TgapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.tgaps.TgapsMetric>`_

`interNightBatch <lsst.sims.maf.batches.html#lsst.sims.maf.batches.timeBatch.interNight>`_
--------------------------------------------------------------------------------------------
Metrics included:

  -  All proposals, and WFD+NES Only
  -  `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
  - `NightgapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.tgaps.NightgapsMetric>`_
  - `Median InterNightGapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.InterNightGapsMetric>`_ (Per filter and all filters)
  - `Maximum InterNightGapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.InterNightGapsMetric>`_ (Per filter and all filters)

Metadata metrics
================

`metadataBatch <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.metadataBatch>`_
----------------------------------------------------------------------------------------

A large set of metrics about the metadata of each visit:

 - distributions of airmass
 - normalized airmass
 - seeing
 - sky brightness
 - single visit depth
 - hour angle
 - distance to the moon
 - solar elongation

Metrics included:

 -  WFD and all proposals
 -  Metrics are calculated for all and a per filter basis.
 -  Summaries of all values using `UniSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.UniSlicer>`_
 -  Metic value on sky using `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
 - `CountMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountMetric>`_
 - `MedianMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MedianMetric>`_
 - `MaxMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MaxMetric>`_
