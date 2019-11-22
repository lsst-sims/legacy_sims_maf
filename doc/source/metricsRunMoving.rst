================================================
Metrics in run_moving.py script
================================================

Running moving object metrics has been set up as a multi-step process. 
The reason for this change (from a single script) is that it is often most
convenient (and faster!) to split a larger population into smaller subsets; 
sims_movingObjects can be run on each of these subsets in parallel, the
first step of metric calculation can be run in parallel - and then, with 
these results in hand, the metric results can be joined back together and then 
the summary metrics which require the full population can be calculated.

Thus metric calculation can be most quickly done as follows: 
* take a complete solar system population and split it into (ten) subsets, where each
subset is a standard input file for makeLSSTobs.py (in sims_movingObjects). If the original complete
population file is SSOpop, the individual subset files should be named SSOpop_N, where N 
can range from 0-9 (this could be modified by changing run_moving_join.py later, but I chose 
to standardize for 10 subsets as the cluster I usually run these on has 10 cpus per node). 
* run makeLSSTobs.py for each of the subset populations. Here you only need to refer to the 
subset file -- i.e. makeLSSTobs.py --orbitFile SSOpop_N ..
* run the first step of metric calculation: run_moving_calc.py. The IMPORTANT thing to note here
is that in each of the MAF metric steps, you should refer to the complete population file, not the subset.
ie. run_moving_calc.py --orbitFile SSOpop --obsFile SSOpop_N_obs.txt
By using the entire population file as the orbit file, the metric results will be stored in their
proper places in a metric array. Make sure that the H-ranges specified are the same for all of the subsets.
* run the second step to combine the outputs of each of the calculated metrics, into a single metric 
output file that contains all of the subsets: run_moving_join.py. The script will automatically look
for directories which match the expected pattern for subsets, and then create a single (new) output 
directory that contains the joined files.
* run the final step to calculate summary metrics across the entire population, using these joined
metric output files.

If you do not want to subset the input population file, it is still recommended to run each of the
MAF evaluation steps .. the run_moving_join.py script will basically be a no-op.


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

