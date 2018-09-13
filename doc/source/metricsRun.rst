============================
Metrics in run_all.py script
============================
The `run_all.py` script included in MAF runs a very large number of useful
metrics covering metadata about observing history, survey performance, and
those from the LSST SRD. Here we will list and summarize the metics included
in this script.



`SRD metrics <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.srdBatch>`_
==================================================================================

`fOBatch <lsst.sims.maf.batches.html#lsst.sims.maf.batches.srdBatch.fOBatch>`_
-------------------------------------------------------------------------------

    Metric Info:

        - WFD only, and all proposals
        - `fOArea <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.fOArea>`_
        - `fONv <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.fONv>`_

`astrometryBatch <lsst.sims.maf.batches.html#lsst.sims.maf.batches.srdBatch.astrometryBatch>`_
------------------------------------------------------------------------------------------------

    Metric Info:

        -  WFD only, and all proposals
        - `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
        - `ProperMotionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ProperMotionMetric>`_
        - `ParallaxMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxMetric>`_
        - `ParallaxCoverageMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxCoverageMetric>`_
        - `ParallaxDcrDegenMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxDcrDegenMetric>`_


`rapidRevisitBatch <lsst.sims.maf.batches.html#lsst.sims.maf.batches.srdBatch.rapidRevisitBatch>`_
---------------------------------------------------------------------------------------------------

    Metric Info:

        -  WFD only, and all proposals
        -  `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
        - `RapidRevisitMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.RapidRevisitMetric>`_


`Cadence metrics (pairs and time) <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.timeBatch>`_
=======================================================================================================

`intraNight <lsst.sims.maf.batches.html#lsst.sims.maf.batches.timeBatch.intraNight>`_
--------------------------------------------------------------------------------------

    - Metric Info:

        -  All proposals, and WFD+NES Only
        -  `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
        - `PairFractionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.visitGroupsMetric.PairFractionMetric>`_
        - `NRevisitsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.NRevisitsMetric>`_
        - `Median IntraNightGapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.IntraNightGapsMetric>`_
        - `NVisitsPerNightMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.tgaps.NVisitsPerNightMetric>`_
        - `TgapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.tgaps.TgapsMetric>`_

`interNight <lsst.sims.maf.batches.html#lsst.sims.maf.batches.timeBatch.interNight>`_
--------------------------------------------------------------------------------------

    - Metric Info:

        -  All proposals, and WFD+NES Only
        -  `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
        - `NightgapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.tgaps.NightgapsMetric>`_
        - `Median InterNightGapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.InterNightGapsMetric>`_ (Per filter and all filters)
        - `Maximum InterNightGapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.InterNightGapsMetric>`_ (Per filter and all filters)


`Metadata metrics <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.metadataBatch>`_
===========================================================================================

`allMetadata <lsst.sims.maf.batches.html#lsst.sims.maf.batches.metadataBatch.allMetadata>`_
--------------------------------------------------------------------------------------------

    A large set of metrics about the metadata of each visit:

        - distributions of airmass
        - normalized airmass
        - seeing
        - sky brightness
        - single visit depth
        - hour angle
        - distance to the moon
        - solar elongation

    Metric Info:

        -  WFD and all proposals
        -  Metrics are calculated for all and a per filter basis.
        -  Summaries of all values using `UniSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.UniSlicer>`_
        -  Metic value on sky using `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
        - `CountMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountMetric>`_
        - `MedianMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MedianMetric>`_
        - `MaxMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MaxMetric>`_

`NVisits, CoaddedM5, Teff <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.visitdepthBatch>`_
======================================================================================================

`nvisitsM5Maps <lsst.sims.maf.batches.html#lsst.sims.maf.batches.visitdepthBatch.nvisitsM5Maps>`_
--------------------------------------------------------------------------------------------------

    Metric Info:

      -  All proposals, and WFD Only
      - `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
      - `CountMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountMetric>`_ (Per filter and all filters)
      - `Coaddm5Metric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.Coaddm5Metric>`_ (Per filter and all filters)

`tEffMetrics <lsst.sims.maf.batches.html#lsst.sims.maf.batches.visitdepthBatch.tEffMetrics>`_
----------------------------------------------------------------------------------------------

    Metric Info:

      -  All proposals, and WFD Only
      - `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
      - `TeffMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.TeffMetric>`_ (Per filter and all filters)

`nvisitsPerNight <lsst.sims.maf.batches.html#lsst.sims.maf.batches.visitdepthBatch.nvisitsPerNight>`_
------------------------------------------------------------------------------------------------------

    Metric Info:

      -  All proposals, and WFD Only
      - `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
      - `CountMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountMetric>`_ (Per filter and all filters)

`nvisitsPerProp <lsst.sims.maf.batches.html#lsst.sims.maf.batches.visitdepthBatch.nvisitsPerNight>`_
-----------------------------------------------------------------------------------------------------

    Metric Info:

      -  All proposals, and WFD Only
      - `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
      - `CountMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountMetric>`_ (Per filter and all filters)


`Alt/Az NVisits <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.altazBatch>`_
=======================================================================================

`altazLambert <lsst.sims.maf.batches.html#lsst.sims.maf.batches.altazBatch.altazLambert>`_
-------------------------------------------------------------------------------------------

    Metric Info:

        -  Per filter and all filters
        - `healpixSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.healpixSlicer>`_
        - `CountMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountMetric>`_

`Slew metrics <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.slewBatch>`_
====================================================================================

`slewBasics <lsst.sims.maf.batches.html#lsst.sims.maf.batches.slewBatch.slewBasics>`_
--------------------------------------------------------------------------------------

    Metric Info:

        -  Slew times and slew distances
        - `UniSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.UniSlicer>`_
        - `CountMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountMetric>`_

`Open shutter metrics <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.openshutterBatch>`_
===================================================================================================

`openshutterFractions <lsst.sims.maf.batches.html#lsst.sims.maf.batches.openshutterBatch.openshutterFractions>`_
-----------------------------------------------------------------------------------------------------------------

    Metric Info:

        -  Per night and whole survey
        - `UniSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.UniSlicer>`_
        - `OpenShutterFractionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.OpenShutterFractionMetric>`_
        - `CountUniqueMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountUniqueMetric>`_
        - `FullRangeMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.FullRangeMetric>`_

`Per night and whole survey filter changes <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.filterchangeBatch>`_
==========================================================================================================================

`filtersPerNight <lsst.sims.maf.batches.html#lsst.sims.maf.batches.filterchangeBatch.filtersPerNight>`_
---------------------------------------------------------------------------------------------------------

    Metric Info:

        - `OneDSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.OneDSlicer>`_
        - `NChangesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.NChangesMetric>`_
        - `MinTimeBetweenStatesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.MinTimeBetweenStatesMetric>`_
        - `NStateChangesFasterThanMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.NStateChangesFasterThanMetric>`_
        - `MaxStateChangesWithinMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.MaxStateChangesWithinMetric>`_

`filtersWholeSurvey <lsst.sims.maf.batches.html#lsst.sims.maf.batches.filterchangeBatch.filtersWholeSurvey>`_
--------------------------------------------------------------------------------------------------------------

    Metric Info:

        - `UniSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.UniSlicer>`_
        - `NChangesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.NChangesMetric>`_
        - `MinTimeBetweenStatesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.MinTimeBetweenStatesMetric>`_
        - `NStateChangesFasterThanMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.NStateChangesFasterThanMetric>`_
        - `MaxStateChangesWithinMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.MaxStateChangesWithinMetric>`_


`Hourglass plots <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.hourglassBatch>`_
============================================================================================

`hourglassPlots <lsst.sims.maf.batches.html#lsst.sims.maf.batches.hourglassBatch.hourglassPlots>`_
---------------------------------------------------------------------------------------------------

    Metric Info:

        - `HourglassSlicer <lsst.sims.maf.slicers.html#module-lsst.sims.maf.slicers.HourglassSlicer>`_
        - `HourglassMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.hourglassMetric.HourglassMetric>`_
