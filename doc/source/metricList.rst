=================
Available metrics
=================
Core LSST MAF metrics
=====================
 
- `AbsMaxMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.AbsMaxMetric>`_ 
 	 Calculate the max of the absolute value of a simData column slice.
- `AbsMaxPercentMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.AbsMaxPercentMetric>`_ 
 	 Return the percent of the data which has the absolute value of the max value of the data.
- `AbsMeanMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.AbsMeanMetric>`_ 
 	 Calculate the mean of the absolute value of a simData column slice.
- `AbsMedianMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.AbsMedianMetric>`_ 
 	 Calculate the median of the absolute value of a simData column slice.
- `AccumulateCountMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.AccumulateCountMetric>`_ 
 	 Calculate the accumulated stat
- `AccumulateM5Metric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.AccumulateM5Metric>`_ 
 	 Calculate the accumulated stat
- `AccumulateMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.AccumulateMetric>`_ 
 	 Calculate the accumulated stat
- `AccumulateUniformityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.AccumulateUniformityMetric>`_ 
 	 Make a 2D version of UniformityMetric
- `ActivityOverPeriodMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ActivityOverPeriodMetric>`_ 
 	 Count the fraction of the orbit (when split into nBins) that receive
- `ActivityOverTimeMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ActivityOverTimeMetric>`_ 
 	 Count the time periods where we would have a chance to detect activity on
- `AveGapMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.AveGapMetric>`_ 
 	 Calculate the gap between consecutive observations, in hours.
- `AveSlewFracMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.slewMetrics.AveSlewFracMetric>`_ 
 	 Base class for the metrics.
- `BaseMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.baseMetric.BaseMetric>`_ 
 	 Base class for the metrics.
- `BaseMoMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.BaseMoMetric>`_ 
 	 Base class for the moving object metrics.
- `BinaryMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.BinaryMetric>`_ 
 	 Return 1 if there is data. 
- `BruteOSFMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.BruteOSFMetric>`_ 
 	 Assume I can't trust the slewtime or visittime colums.
- `ChipVendorMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.chipVendorMetric.ChipVendorMetric>`_ 
 	 See what happens if we have chips from different vendors
- `Coaddm5Metric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.Coaddm5Metric>`_ 
 	 Calculate the coadded m5 value at this gridpoint.
- `ColorDeterminationMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ColorDeterminationMetric>`_ 
 	 Identify objects which could have observations suitable to determine colors.
- `CompletenessMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.CompletenessMetric>`_ 
 	 Compute the completeness and joint completeness 
- `CountMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountMetric>`_ 
 	 Count the length of a simData column slice. 
- `CountRatioMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountRatioMetric>`_ 
 	 Count the length of a simData column slice, then divide by 'normVal'. 
- `CountSubsetMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountSubsetMetric>`_ 
 	 Count the length of a simData column slice which matches 'subset'. 
- `CountUniqueMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.CountUniqueMetric>`_ 
 	 Return the number of unique values.
- `CrowdingMagUncertMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.crowdingMetric.CrowdingMagUncertMetric>`_ 
 	 Given a stellar magnitude, calculate the mean uncertainty on the magnitude from crowding.
- `CrowdingMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.crowdingMetric.CrowdingMetric>`_ 
 	 Calculate whether the coadded depth in r has exceeded the confusion limit
- `DiscoveryChancesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.DiscoveryChancesMetric>`_ 
 	 Count the number of discovery opportunities for an object.
- `DiscoveryMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.DiscoveryMetric>`_ 
 	 Identify the discovery opportunities for an object.
- `Discovery_EcLonLatMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_EcLonLatMetric>`_ 
 	 Returns the ecliptic lon/lat and solar elongation (in degrees) of the i-th discovery opportunity.
- `Discovery_N_ChancesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_N_ChancesMetric>`_ 
 	 Child metric to be used with DiscoveryMetric.
- `Discovery_N_ObsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_N_ObsMetric>`_ 
 	 Calculates the number of observations in the i-th discovery track.
- `Discovery_RADecMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_RADecMetric>`_ 
 	 Returns the RA/Dec of the i-th discovery opportunity.
- `Discovery_TimeMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_TimeMetric>`_ 
 	 Returns the time of the i-th discovery opportunity.
- `Discovery_VelocityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.Discovery_VelocityMetric>`_ 
 	 Returns the sky velocity of the i-th discovery opportunity.
- `ExgalM5 <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.exgalM5.ExgalM5>`_ 
 	 Calculate co-added five-sigma limiting depth after dust extinction.
- `FftMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.fftMetric.FftMetric>`_ 
 	 Calculate a truncated FFT of the exposure times.
- `FilterColorsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.FilterColorsMetric>`_ 
 	 Calculate an RGBA value that accounts for the filters used up to time t0.
- `FracAboveMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.FracAboveMetric>`_ 
 	 Find the fraction of data values above a given value.
- `FracBelowMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.FracBelowMetric>`_ 
 	 Find the fraction of data values below a given value.
- `FullRangeAngleMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.FullRangeAngleMetric>`_ 
 	 Calculate the full range of an angular (degrees) simData column slice.
- `FullRangeMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.FullRangeMetric>`_ 
 	 Calculate the range of a simData column slice.
- `HighVelocityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.HighVelocityMetric>`_ 
 	 Count the number of times an asteroid is observed with a velocity high enough to make it appear
- `HighVelocityNightsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.HighVelocityNightsMetric>`_ 
 	 Count the number of times an asteroid is observed with a velocity high enough to make it appear
- `HistogramM5Metric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.HistogramM5Metric>`_ 
 	 Calculate the coadded depth for each bin (e.g., per night).
- `HistogramMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.vectorMetrics.HistogramMetric>`_ 
 	 A wrapper to stats.binned_statistic
- `HourglassMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.hourglassMetric.HourglassMetric>`_ 
 	 Plot the filters used as a function of time. Must be used with the Hourglass Slicer.
- `IdentityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.IdentityMetric>`_ 
 	 Return the metric value itself .. this is primarily useful as a summary statistic for UniSlicer metrics.
- `InterNightGapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.InterNightGapsMetric>`_ 
 	 Calculate the gap between consecutive observations between nights, in days.
- `IntraNightGapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.IntraNightGapsMetric>`_ 
 	 Calculate the gap between consecutive observations within a night, in hours.
- `KnownObjectsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.KnownObjectsMetric>`_ 
 	 Identify objects which could be classified as 'previously known' based on their peak V magnitude,
- `LightcurveInversionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.LightcurveInversionMetric>`_ 
 	 Identify objects which would have observations suitable to do lightcurve inversion.
- `LongGapAGNMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.longGapAGNMetric.LongGapAGNMetric>`_ 
 	 max delta-t and average of the top-10 longest gaps.
- `MagicDiscoveryMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.MagicDiscoveryMetric>`_ 
 	 Count the number of discovery opportunities with very good software.
- `MaxMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MaxMetric>`_ 
 	 Calculate the maximum of a simData column slice.
- `MaxPercentMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MaxPercentMetric>`_ 
 	 Return the percent of the data which has the maximum value.
- `MaxStateChangesWithinMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.MaxStateChangesWithinMetric>`_ 
 	 Compute the maximum number of changes of state that occur within a given timespan.
- `MeanAngleMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MeanAngleMetric>`_ 
 	 Calculate the mean of an angular (degree) simData column slice.
- `MeanMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.MeanMetric>`_ 
 	 Calculate the mean of a simData column slice.
- `MeanValueAtHMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moSummaryMetrics.MeanValueAtHMetric>`_ 
 	 Return the mean value of a metric at a given H.
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
 	 Count the number of distinct nights an object is observed.
- `NObsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.NObsMetric>`_ 
 	 Count the total number of observations where an object was 'visible'.
- `NObsNoSinglesMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.NObsNoSinglesMetric>`_ 
 	 Count the number of observations for an object, but don't
- `NRevisitsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.NRevisitsMetric>`_ 
 	 Calculate the number of consecutive visits with time differences less than dT.
- `NStateChangesFasterThanMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.NStateChangesFasterThanMetric>`_ 
 	 Compute the number of changes of state that happen faster than 'cutoff'.
- `NightPointingMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.nightPointingMetric.NightPointingMetric>`_ 
 	 Gather relevant information for a night to plot.
- `NormalizeMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.NormalizeMetric>`_ 
 	 Return a metric values divided by 'normVal'. Useful for turning summary statistics into fractions.
- `NoutliersNsigmaMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.NoutliersNsigmaMetric>`_ 
 	 Calculate the # of visits less than nSigma below the mean (nSigma<0) or
- `ObsArcMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.ObsArcMetric>`_ 
 	 Calculate the difference between the first and last observation of an object.
- `OpenShutterFractionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.OpenShutterFractionMetric>`_ 
 	 Compute the fraction of time the shutter is open compared to the total time spent observing.
- `OptimalM5Metric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.optimalM5Metric.OptimalM5Metric>`_ 
 	 Compare the co-added depth of the survey to one where
- `PairFractionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.visitGroupsMetric.PairFractionMetric>`_ 
 	 What fraction of observations are part of a pair.
- `PairMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.pairMetric.PairMetric>`_ 
 	 Count the number of pairs that could be used for Solar System object detection
- `ParallaxCoverageMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxCoverageMetric>`_ 
 	 Check how well the parallax factor is distributed. Subtracts the weighted mean position of the
- `ParallaxDcrDegenMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxDcrDegenMetric>`_ 
 	 Use the full parallax and DCR displacement vectors to find if they are degenerate.
- `ParallaxMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ParallaxMetric>`_ 
 	 Calculate the uncertainty in a parallax measurement given a series of observations.
- `PassMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.PassMetric>`_ 
 	 Just pass the entire array through
- `PeakVMagMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moMetrics.PeakVMagMetric>`_ 
 	 Pull out the peak V magnitude of all observations of the object.
- `PercentileMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.PercentileMetric>`_ 
 	 Find the value of a column at a given percentile.
- `PhaseGapMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.phaseGapMetric.PhaseGapMetric>`_ 
 	 Measure the maximum gap in phase coverage for observations of periodic variables.
- `ProperMotionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.ProperMotionMetric>`_ 
 	 Calculate the uncertainty in the returned proper motion.  Assuming Gaussian errors.
- `RadiusObsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.calibrationMetrics.RadiusObsMetric>`_ 
 	 find the radius in the focal plane. returns things in degrees.
- `RapidRevisitMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.RapidRevisitMetric>`_ 
 	 Calculate uniformity of time between consecutive visits on short timescales (for RAV1).
- `RmsAngleMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.RmsAngleMetric>`_ 
 	 Calculate the standard deviation of an angular (degrees) simData column slice.
- `RmsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.RmsMetric>`_ 
 	 Calculate the standard deviation of a simData column slice.
- `RobustRmsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.RobustRmsMetric>`_ 
 	 Use the inter-quartile range of the data to estimate the RMS.  
- `SlewContributionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.slewMetrics.SlewContributionMetric>`_ 
 	 Base class for the metrics.
- `StarDensityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.starDensity.StarDensityMetric>`_ 
 	 Interpolate the stellar luminosity function to return the number of
- `SumMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.SumMetric>`_ 
 	 Calculate the sum of a simData column slice.
- `TableFractionMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.TableFractionMetric>`_ 
 	 Count the completeness (for many fields) and summarize how many fields have given completeness levels
- `TeffMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.technicalMetrics.TeffMetric>`_ 
 	 Effective time equivalent for a given set of visits.
- `TemplateExistsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.TemplateExistsMetric>`_ 
 	 Calculate the fraction of images with a previous template image of desired quality.
- `TgapsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.tgaps.TgapsMetric>`_ 
 	 Histogram all the time gaps.
- `TotalPowerMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.TotalPowerMetric>`_ 
 	 Calculate the total power in the angular power spectrum between lmin/lmax.
- `TransientMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.transientMetrics.TransientMetric>`_ 
 	 Calculate what fraction of the transients would be detected. Best paired with a spatial slicer.
- `UniformityMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.cadenceMetrics.UniformityMetric>`_ 
 	 Calculate how uniformly the observations are spaced in time.
- `UniqueRatioMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.simpleMetrics.UniqueRatioMetric>`_ 
 	 Return the number of unique values divided by the total number of values.
- `ValueAtHMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.moSummaryMetrics.ValueAtHMetric>`_ 
 	 Return the metric value at a given H value.
- `VisitGroupsMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.visitGroupsMetric.VisitGroupsMetric>`_ 
 	 Count the number of visits per night within deltaTmin and deltaTmax.
- `ZeropointMetric <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.ZeropointMetric>`_ 
 	 Return a metric values with the addition of 'zp'. Useful for altering the zeropoint for summary statistics.
- `fOArea <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.fOArea>`_ 
 	 Metric to calculate the FO Area.
- `fONv <lsst.sims.maf.metrics.html#lsst.sims.maf.metrics.summaryMetrics.fONv>`_ 
 	 Metric to calculate the FO_Nv.
 
