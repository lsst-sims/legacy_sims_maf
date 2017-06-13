=================
Available metrics
=================
Core LSST MAF metrics
=====================
 
- `AccumulateCountMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.AccumulateCountMetric>`_ 
 	 Calculate the number of visits over time.
- `AccumulateM5Metric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.AccumulateM5Metric>`_ 
 	 The 5-sigma depth accumulated over time.
- `AccumulateMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.AccumulateMetric>`_ 
 	 Calculate the accumulated stat.
- `AccumulateUniformityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.AccumulateUniformityMetric>`_ 
 	 Make a 2D version of UniformityMetric.
- `ActivityOverPeriodMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ActivityOverPeriodMetric>`_ 
 	 Count the fraction of the orbit that receive observations, such that activity is detectable.
- `ActivityOverTimeMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ActivityOverTimeMetric>`_ 
 	 Count the time periods where we would have a chance to detect activity on a moving object.
- `AveGapMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.AveGapMetric>`_ 
 	 Calculate the gap between consecutive observations, in hours.
- `AveSlewFracMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.slewMetrics.AveSlewFracMetric>`_ 
 	 Average time for slew activity, multiplied by percent of total slews.
- `BaseMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.baseMetric.BaseMetric>`_ 
 	 Base class for the metrics.
- `BaseMoMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.BaseMoMetric>`_ 
 	 Base class for the moving object metrics.
- `BinaryMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.BinaryMetric>`_ 
 	 Return 1 if there is data.
- `ChipVendorMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.chipVendorMetric.ChipVendorMetric>`_ 
 	 Examine coverage with a mixed chip vendor focal plane.
- `Coaddm5Metric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.Coaddm5Metric>`_ 
 	 Calculate the coadded m5 value at this gridpoint.
- `ColorDeterminationMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ColorDeterminationMetric>`_ 
 	 Identify SS objects which could have observations suitable to determine colors.
- `CompletenessMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.CompletenessMetric>`_ 
 	 Compute the completeness and joint completeness of requested observations.
- `CountMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountMetric>`_ 
 	 Count the length of a simData column slice.
- `CountRatioMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountRatioMetric>`_ 
 	 Count the length of a simData column slice, then divide by a normalization value.
- `CountSubsetMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountSubsetMetric>`_ 
 	 Count the length of a simData column slice which matches 'subset'.
- `CountUniqueMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountUniqueMetric>`_ 
 	 Return the number of unique values
- `CrowdingMagUncertMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.crowdingMetric.CrowdingMagUncertMetric>`_ 
 	 Given a stellar magnitude, calculate the mean uncertainty on the magnitude from crowding.
- `CrowdingMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.crowdingMetric.CrowdingMetric>`_ 
 	 Calculate whether the coadded depth in r has exceeded the confusion limit.
- `DiscoveryChancesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.DiscoveryChancesMetric>`_ 
 	 Count the number of discovery opportunities for an SS object.
- `DiscoveryMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.DiscoveryMetric>`_ 
 	 Identify the discovery opportunities for an SS object.
- `Discovery_EcLonLatMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_EcLonLatMetric>`_ 
 	 Returns the ecliptic lon/lat and solar elongation (deg) of the i-th SS object discovery opportunity.
- `Discovery_N_ChancesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_N_ChancesMetric>`_ 
 	 Count the number of discovery opportunities for SS object in a window between nightStart/nightEnd.
- `Discovery_N_ObsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_N_ObsMetric>`_ 
 	 Calculates the number of observations of SS object in the i-th discovery track.
- `Discovery_RADecMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_RADecMetric>`_ 
 	 Returns the RA/Dec of the i-th SS object discovery opportunity.
- `Discovery_TimeMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_TimeMetric>`_ 
 	 Returns the time of the i-th SS object discovery opportunity.
- `Discovery_VelocityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_VelocityMetric>`_ 
 	 Returns the sky velocity of the i-th SS object discovery opportunity.
- `ExgalM5 <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.exgalM5.ExgalM5>`_ 
 	 Calculate co-added five-sigma limiting depth after dust extinction.
- `FftMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.fftMetric.FftMetric>`_ 
 	 Calculate a truncated FFT of the exposure times.
- `FilterColorsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.FilterColorsMetric>`_ 
 	 Calculate an RGBA value that accounts for the filters used up to time t0.
- `FracAboveMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.FracAboveMetric>`_ 
 	 Find the fraction above a certain value.
- `FracBelowMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.FracBelowMetric>`_ 
 	 Find the fraction below a certain value.
- `FullRangeAngleMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.FullRangeAngleMetric>`_ 
 	 Calculate the full range of an angular (radians) simData column slice.
- `FullRangeMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.FullRangeMetric>`_ 
 	 Calculate the range of a simData column slice.
- `HighVelocityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.HighVelocityMetric>`_ 
 	 Count the number of observations with high velocities.
- `HighVelocityNightsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.HighVelocityNightsMetric>`_ 
 	 Count the number of nights with nObsPerNight number of high velocity detections.
- `HistogramM5Metric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.HistogramM5Metric>`_ 
 	 Calculate the coadded depth for each bin (e.g., per night).
- `HistogramMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.HistogramMetric>`_ 
 	 A wrapper to stats.binned_statistic.
- `HourglassMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.hourglassMetric.HourglassMetric>`_ 
 	 Plot the filters used as a function of time.
- `IdentityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.IdentityMetric>`_ 
 	 Return the input value itself.
- `InterNightGapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.InterNightGapsMetric>`_ 
 	 Calculate the gap between consecutive observations between nights, in days.
- `IntraNightGapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.IntraNightGapsMetric>`_ 
 	 Calculate the gap between consecutive observations within a night, in hours.
- `KnownObjectsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.KnownObjectsMetric>`_ 
 	 Identify SS objects which could be classified as 'previously known' based on their peak V magnitude.
- `LightcurveInversionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.LightcurveInversionMetric>`_ 
 	 Identify SS objects which would have observations suitable to do lightcurve inversion.
- `LongGapAGNMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.longGapAGNMetric.LongGapAGNMetric>`_ 
 	 Compute the max delta-t and average of the top-10 longest observation gaps.
- `MagicDiscoveryMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.MagicDiscoveryMetric>`_ 
 	 Count the number of SS object discovery opportunities with very good software.
- `MaxMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MaxMetric>`_ 
 	 Calculate the maximum of a simData column slice.
- `MaxPercentMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MaxPercentMetric>`_ 
 	 Return the percent of the data which has the maximum value.
- `MaxStateChangesWithinMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.MaxStateChangesWithinMetric>`_ 
 	 Compute the maximum number of changes of state that occur within a given timespan.
- `MeanAngleMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MeanAngleMetric>`_ 
 	 Calculate the mean of an angular (radians) simData column slice.
- `MeanMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MeanMetric>`_ 
 	 Calculate the mean of a simData column slice.
- `MeanValueAtHMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moSummaryMetrics.MeanValueAtHMetric>`_ 
 	 Return the mean value of a metric at a given H.
- `MedianAbsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MedianAbsMetric>`_ 
 	 Calculate the median of the absolute value of a simData column slice.
- `MedianMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MedianMetric>`_ 
 	 Calculate the median of a simData column slice.
- `MetricRegistry <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.baseMetric.MetricRegistry>`_ 
 	 Meta class for metrics, to build a registry of metric classes.
- `MinMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MinMetric>`_ 
 	 Calculate the minimum of a simData column slice.
- `MinTimeBetweenStatesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.MinTimeBetweenStatesMetric>`_ 
 	 Compute the minimum time between changes of state in a column value.
- `MoCompletenessAtTimeMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moSummaryMetrics.MoCompletenessAtTimeMetric>`_ 
 	 Calculate the completeness (relative to the entire population) <= a given H as a function of time,
- `MoCompletenessMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moSummaryMetrics.MoCompletenessMetric>`_ 
 	 Calculate the completeness (relative to the entire population), given the counts of discovery chances.
- `NChangesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.NChangesMetric>`_ 
 	 Compute the number of times a column value changes.
- `NNightsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.NNightsMetric>`_ 
 	 Count the number of distinct nights an SS object is observed.
- `NObsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.NObsMetric>`_ 
 	 Count the total number of observations where an SS object was 'visible'.
- `NObsNoSinglesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.NObsNoSinglesMetric>`_ 
 	 Count the number of observations for an SS object, but not if it was a single observation on a night.
- `NRevisitsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.NRevisitsMetric>`_ 
 	 Calculate the number of (consecutive) visits with time differences less than dT.
- `NStateChangesFasterThanMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.NStateChangesFasterThanMetric>`_ 
 	 Compute the number of changes of state that happen faster than 'cutoff'.
- `NightPointingMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.nightPointingMetric.NightPointingMetric>`_ 
 	 Gather relevant information for a single night to plot.
- `NormalizeMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.NormalizeMetric>`_ 
 	 Return a metric values divided by 'normVal'.
- `NoutliersNsigmaMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.NoutliersNsigmaMetric>`_ 
 	 Calculate the number of visits outside the given sigma threshold.
- `ObsArcMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ObsArcMetric>`_ 
 	 Calculate the time difference between the first and last observation of an SS object.
- `OpenShutterFractionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.OpenShutterFractionMetric>`_ 
 	 Compute the fraction of time the shutter is open compared to the total time the dome is open.
- `OptimalM5Metric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.optimalM5Metric.OptimalM5Metric>`_ 
 	 Compare the co-added depth of the survey to one where all the observations were taken on the meridian.
- `PairMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.pairMetric.PairMetric>`_ 
 	 Count the number of pairs that could be used for Solar System object detection.
- `ParallaxCoverageMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxCoverageMetric>`_ 
 	 Check how well the parallax factor is distributed.
- `ParallaxDcrDegenMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxDcrDegenMetric>`_ 
 	 Compute parallax and DCR displacement vectors to find if they are degenerate.
- `ParallaxMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxMetric>`_ 
 	 Calculate the uncertainty in a parallax measures given a serries of observations.
- `PassMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.PassMetric>`_ 
 	 Just pass the entire array.
- `PeakVMagMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.PeakVMagMetric>`_ 
 	 Pull out the peak V magnitude of all observations of an SS object.
- `PercentileMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.PercentileMetric>`_ 
 	 Find the value of a column at a given percentile.
- `PhaseGapMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.phaseGapMetric.PhaseGapMetric>`_ 
 	 Measure the maximum gap in phase coverage for observations of periodic variables.
- `ProperMotionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ProperMotionMetric>`_ 
 	 Calculate the uncertainty in the fitted proper motion assuming Gaussian errors.
- `RadiusObsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.RadiusObsMetric>`_ 
 	 Find the radius a point falls in the focal plane.
- `RapidRevisitMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.RapidRevisitMetric>`_ 
 	 Calculate uniformity of time between consecutive visits on short timescales (for RAV1).
- `RmsAngleMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.RmsAngleMetric>`_ 
 	 Calculate the standard deviation of an angular (radians) simData column slice.
- `RmsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.RmsMetric>`_ 
 	 Calculate the standard deviation of a simData column slice.
- `RobustRmsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.RobustRmsMetric>`_ 
 	 Use the inter-quartile range of the data to estimate the RMS.
- `SlewContributionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.slewMetrics.SlewContributionMetric>`_ 
 	 Average time a slew activity is in the critical path.
- `StarDensityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.starDensity.StarDensityMetric>`_ 
 	 Interpolate the stellar luminosity function to return the number of
- `SumMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.SumMetric>`_ 
 	 Calculate the sum of a simData column slice.
- `TableFractionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.TableFractionMetric>`_ 
 	 Compute a table for the completeness of requested observations.
- `TeffMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.TeffMetric>`_ 
 	 Effective exposure time for a given set of visits based on fiducial 5-sigma depth expectations.
- `TemplateExistsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.TemplateExistsMetric>`_ 
 	 Calculate the fraction of images with a previous template image of desired quality.
- `TgapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.tgaps.TgapsMetric>`_ 
 	 Histogram up all the time gaps.
- `TotalPowerMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.TotalPowerMetric>`_ 
 	 Calculate the total power in the angular power spectrum between lmin/lmax.
- `TransientMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.transientMetrics.TransientMetric>`_ 
 	 Calculate what fraction of the transients would be detected.
- `UniformityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.UniformityMetric>`_ 
 	 Calculate how uniformly observations are spaced in time.
- `UniqueRatioMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.UniqueRatioMetric>`_ 
 	 Return the number of unique values divided by the total
- `ValueAtHMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moSummaryMetrics.ValueAtHMetric>`_ 
 	 Return the metric value at a given H value.
- `VisitGroupsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.visitGroupsMetric.VisitGroupsMetric>`_ 
 	 Count the number of visits per night within deltaTmin and deltaTmax.
- `ZeropointMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.ZeropointMetric>`_ 
 	 Return a metric values with the addition of zeropoint.
- `fOAreaMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.fOAreaMetric>`_ 
 	 Metric to calculate the FO Area.
- `fONvMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.fONvMetric>`_ 
 	 Metric to calculate the FO_Nv.
 
Contributed mafContrib metrics
==============================
 
- `AngularSpreadMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/angularSpread.py>`_ 
  	 Compute the angular spread statistic which measures uniformity of a distribution angles
- `CampaignLengthMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/campaignLengthMetric.py>`_ 
  	 The campaign length, in seasons. In the main survey this is 
- `GRBTransientMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/GRBTransientMetric.py>`_ 
  	 Detections for on-axis GRB afterglows decaying as 
- `GalaxyCountsMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/lssMetrics.py>`_ 
  	 Estimate the number of galaxies expected at a particular coadded depth.
- `MeanNightSeparationMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/meanNightSeparationMetric.py>`_ 
  	 The mean separation between nights within a season, and then 
- `NumObsMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/numObsMetric.py>`_ 
  	 Calculate the number of observations per data slice, e.g. HealPix pixel when using HealPix slicer.
- `PeriodDeviationMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/varMetrics.py>`_ 
  	 Measure the percentage deviation of recovered periods for
- `PeriodMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/PeriodicMetric.py>`_ 
  	 From a set of observation times, uses code provided by Robert Siverd (LCOGT) to calculate the spectral window function.
- `PeriodicStarMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/periodicStarMetric.py>`_ 
  	 At each slicePoint, run a Monte Carlo simulation to see how well a periodic source can be fit.
- `RelRmsMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/photPrecMetrics.py>`_ 
  	 Base class for the metrics.
- `SEDSNMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/photPrecMetrics.py>`_ 
  	 Computes the S/Ns for a given SED 
- `SNMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/photPrecMetrics.py>`_ 
  	 Calculate the signal to noise metric in a given filter for an object 
- `SeasonLengthMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/seasonLengthMetric.py>`_ 
  	 The mean season length, in months. The SeasonStacker must be run 
- `StarCountMassMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/StarCountMassMetric.py>`_ 
  	 Find the number of stars in a given field in the mass range fainter than magnitude 16 and bright enough to have noise less than 0.03 in a given band. M1 and M2 are the upper and lower limits of the mass range. 'band' is the band to be observed.
- `StarCountMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/StarCountMetric.py>`_ 
  	 Find the number of stars in a given field between D1 and D2 in parsecs.
- `TdcMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/tdcMetric.py>`_ 
  	 Base class for the metrics.
- `ThreshSEDSNMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/photPrecMetrics.py>`_ 
  	 Computes the metric whether the S/N is bigger than the threshold
- `TransientAsciiMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/transientAsciiMetric.py>`_ 
  	 Based on the transientMetric, but uses an ascii input file and provides option to write out lightcurve.
- `TripletBandMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/TripletMetric.py>`_ 
  	 Find the number of 'triplets' of three images taken in the same band, based on user-selected minimum and maximum intervals (in hours),
- `TripletMetric <http://github.com/lsst-nonproject/sims_maf_contrib/tree/master/mafContrib/TripletMetric.py>`_ 
  	 Find the number of 'triplets' of three images taken in any band, based on user-selected minimum and maximum intervals (in hours),
 
