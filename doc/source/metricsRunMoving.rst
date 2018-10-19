================================================
Metrics in run_moving.py script
================================================
The `run_moving.py` script runs a number of solar system object oriented metrics,
and requires an input SSO observation file to run (e.g. you must generate this
observation file using something like sims_movingObjects `makeLSSTobs.py` first).


`QuickDiscoveryBatch <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.quickDiscoveryBatch>`_
=====================================================================================================

    The QuickDiscoveryBatch is intended to provide a short but sweet set of discovery metric options.
    It just runs the
    `Discovery Metric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.DiscoveryMetric>`_
    for SNR=5, with discovery criteria of 2 visits / night x 3 nights in a 15 (or 30)
    day window, using whichever version of trailing losses (detection = dmagDetect or trailing =
    dmagTrail) are specified. Please see lsst.sims.movingObjects for more information on detection vs
    trailing losses.

    This will also produce differential and cumulative completeness estimates for the input population,
    both as a function of H magnitude and as a function of time.

    Example of main output files:
        baseline2018b_Discovery_2x3in15_MBAs_3_pairs_in_15_nights_SNReq5_detection_loss_MOOB.npz
        baseline2018b_Discovery_2x3in30_MBAs_3_pairs_in_30_nights_SNReq5_detection_loss_MOOB.npz


    with additional 'child' (or derived) metric output files of:
        baseline2018b_Discovery_N_Chances_MBAs_3_pairs_in_15_nights_SNReq5_detection_loss_MOOB.npz
        baseline2018b_Discovery_N_Chances_MBAs_3_pairs_in_30_nights_SNReq5_detection_loss_MOOB.npz
        baseline2018b_Discovery_Time_MBAs_3_pairs_in_15_nights_SNReq5_detection_loss_MOOB.npz
        baseline2018b_Discovery_Time_MBAs_3_pairs_in_30_nights_SNReq5_detection_loss_MOOB.npz

    and summary metric files of:
      xxxx


`DiscoveryBatch <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.DiscoveryBatch>`_
============================================================================================

    The DiscoveryBatch runs many more discovery metric options, exploring a wide range of discovery criteria.
    It runs the `Discovery Metric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.DiscoveryMetric>`_ looking for:
        * Using a probablistic SNR limit, around SNR=5 (but with a gentle falloff around this value):
            * 2 visits/night, 3 nights within a 15 day window
            * 2 visits/night, 3 nights within a 12 day window
            * 2 visits/night, 3 nights within a 20 day window
            * 2 visits/night, 3 nights within a 25 day window
            * 2 visits/night, 3 nights within a 30 day window
            * 2 visits/night, 4 nights within a 20 day window
            * 3 visits/night, 3 nights within a 30 day window
            * 4 visits/night, 3 nights within a 30 day window
        * Using a SNR=4 cutoff:
            * 2 visits/night, 3 nights within a 15 day window
            * 2 visits/night, 3 nights within a 30 day window
        * Using a SNR=3 cutoff:
            * 2 visits/night, 3 nights within a 15 day window
        * Using a SNR=0 cutoff:
            * 2 visits/night, 3 nights within a 15 day window
        * Using a probabilistic SNR limit, around SNR=5 (with a gentle falloff around that value):
            * Single detections
            * Just a single pair

    Then there are some other discovery metrics, using a probabilistic SNR cutoff (around SNR=5):
        * Detection via trailing (two detections in a night)(`HighVelocityNightsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.HighVelocityNightsMetric>`_)
        * 6 individual detections within a 60 day window (`MagicDiscoveryMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.MagicDiscoveryMetric>`_)


`CharacterizationBatch <lsst.sims.maf.batches.html#module-lsst.sims.maf.batches.CharacterizationBatch>`_
==========================================================================================================

    The characterization batch runs a few metrics intended to shed light on characterization possibilities
    with LSST. These metrics and their rationale are described in more depth in the COSEP.

    * `NObsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.NObsMetric>`_
    * `ObsArcMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ObsArcMetric>`_
    * The `ActivityOverTimeMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ActivityOverTimeMetric>`_ and the `ActivityOverPeriodMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ActivityOverPeriodMetric>`_ are run with a variety of times/periods to identify the likelihood of detecting activity lasting various amounts of time.
    * `LightcurveInversionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.LightcurveInversionMetric>`_
    * `ColorDeterminationMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ColorDeterminationMetric>`_

