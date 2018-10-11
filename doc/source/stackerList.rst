==================
Available stackers
==================
Core LSST MAF stackers
======================
 
- `BaseDitherStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.ditherStackers.BaseDitherStacker>`_ 
 	 Base class for dither stackers.

	 Adds columns: []
- `BaseMoStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.moStackers.BaseMoStacker>`_ 
 	 Base class for moving object (SSobject)  stackers. Relevant for MoSlicer ssObs (pd.dataframe).

	 Adds columns: []
- `BaseStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.baseStacker.BaseStacker>`_ 
 	 Base MAF Stacker: add columns generated at run-time to the simdata array.

	 Adds columns: []
- `DcrStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.generalStackers.DcrStacker>`_ 
 	 Calculate the RA,Dec offset expected for an object due to differential chromatic refraction.

	 Adds columns: ['ra_dcr_amp', 'dec_dcr_amp']
- `EclStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.moStackers.EclStacker>`_ 
 	 Add ecliptic latitude/longitude (ecLat/ecLon) to the slicer ssoObs (in degrees).

	 Adds columns: ['ecLat', 'ecLon']
- `EclipticStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.coordStackers.EclipticStacker>`_ 
 	 Add the ecliptic coordinates of each RA/Dec pointing: eclipLat, eclipLon

	 Adds columns: ['eclipLat', 'eclipLon']
- `FilterColorStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.generalStackers.FilterColorStacker>`_ 
 	 Translate filters ('u', 'g', 'r' ..) into RGB tuples.

	 Adds columns: ['rRGB', 'gRGB', 'bRGB']
- `FiveSigmaStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.generalStackers.FiveSigmaStacker>`_ 
 	 Calculate the 5-sigma limiting depth for a point source in the given conditions.

	 Adds columns: ['m5_simsUtils']
- `GalacticStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.coordStackers.GalacticStacker>`_ 
 	 Add the galactic coordinates of each RA/Dec pointing: gall, galb

	 Adds columns: ['gall', 'galb']
- `HexDitherFieldPerNightStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.ditherStackers.HexDitherFieldPerNightStacker>`_ 
 	 Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.

	 Adds columns: ['hexDitherFieldPerNightRa', 'hexDitherFieldPerNightDec']
- `HexDitherFieldPerVisitStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.ditherStackers.HexDitherFieldPerVisitStacker>`_ 
 	 Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.

	 Adds columns: ['hexDitherFieldPerVisitRa', 'hexDitherFieldPerVisitDec']
- `HexDitherPerNightStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.ditherStackers.HexDitherPerNightStacker>`_ 
 	 Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.

	 Adds columns: ['hexDitherPerNightRa', 'hexDitherPerNightDec']
- `HourAngleStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.generalStackers.HourAngleStacker>`_ 
 	 Add the Hour Angle for each observation.

	 Adds columns: ['HA']
- `M5OptimalStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.m5OptimalStacker.M5OptimalStacker>`_ 
 	 Make a new m5 column as if observations were taken on the meridian.

	 Adds columns: ['m5Optimal']
- `MoMagStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.moStackers.MoMagStacker>`_ 
 	 Add columns relevant to SSobject apparent magnitudes and visibility to the slicer ssoObs

	 Adds columns: ['appMagV', 'appMag', 'SNR', 'vis']
- `NEODistStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.NEODistStacker.NEODistStacker>`_ 
 	 For each observation, find the max distance to a ~144 km NEO,

	 Adds columns: ['MaxGeoDist', 'NEOHelioX', 'NEOHelioY']
- `NFollowStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.nFollowStacker.NFollowStacker>`_ 
 	 Add the number of telescopes ('nObservatories') that could follow up any visit

	 Adds columns: ['nObservatories']
- `NormAirmassStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.generalStackers.NormAirmassStacker>`_ 
 	 Calculate the normalized airmass for each opsim pointing.

	 Adds columns: ['normairmass']
- `OpSimFieldStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.generalStackers.OpSimFieldStacker>`_ 
 	 Add the fieldId of the closest OpSim field for each RA/Dec pointing.

	 Adds columns: ['opsimFieldId']
- `ParallacticAngleStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.generalStackers.ParallacticAngleStacker>`_ 
 	 Add the parallactic angle to each visit.

	 Adds columns: ['PA']
- `ParallaxFactorStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.generalStackers.ParallaxFactorStacker>`_ 
 	 Calculate the parallax factors for each opsim pointing.  Output parallax factor in arcseconds.

	 Adds columns: ['ra_pi_amp', 'dec_pi_amp']
- `RandomDitherFieldPerNightStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.ditherStackers.RandomDitherFieldPerNightStacker>`_ 
 	 Randomly dither the RA and Dec pointings up to maxDither degrees from center,

	 Adds columns: ['randomDitherFieldPerNightRa', 'randomDitherFieldPerNightDec']
- `RandomDitherFieldPerVisitStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.ditherStackers.RandomDitherFieldPerVisitStacker>`_ 
 	 Randomly dither the RA and Dec pointings up to maxDither degrees from center,

	 Adds columns: ['randomDitherFieldPerVisitRa', 'randomDitherFieldPerVisitDec']
- `RandomDitherPerNightStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.ditherStackers.RandomDitherPerNightStacker>`_ 
 	 Randomly dither the RA and Dec pointings up to maxDither degrees from center,

	 Adds columns: ['randomDitherPerNightRa', 'randomDitherPerNightDec']
- `RandomRotDitherPerFilterChangeStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.ditherStackers.RandomRotDitherPerFilterChangeStacker>`_ 
 	 Randomly dither the physical angle of the telescope rotator wrt the mount,

	 Adds columns: ['randomDitherPerFilterChangeRotTelPos']
- `SdssRADecStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.sdssStackers.SdssRADecStacker>`_ 
 	 convert the p1,p2,p3... columns to radians and wrap them 

	 Adds columns: ['RA1', 'Dec1', 'RA2', 'Dec2', 'RA3', 'Dec3', 'RA4', 'Dec4']
- `SeasonStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.generalStackers.SeasonStacker>`_ 
 	 Add an integer label to show which season a given visit is in.

	 Adds columns: ['year', 'season']
- `SpiralDitherFieldPerNightStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.ditherStackers.SpiralDitherFieldPerNightStacker>`_ 
 	 Offset along an equidistant spiral with numPoints, out to a maximum radius of maxDither.

	 Adds columns: ['spiralDitherFieldPerNightRa', 'spiralDitherFieldPerNightDec']
- `SpiralDitherFieldPerVisitStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.ditherStackers.SpiralDitherFieldPerVisitStacker>`_ 
 	 Offset along an equidistant spiral with numPoints, out to a maximum radius of maxDither.

	 Adds columns: ['spiralDitherFieldPerVisitRa', 'spiralDitherFieldPerVisitDec']
- `SpiralDitherPerNightStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.ditherStackers.SpiralDitherPerNightStacker>`_ 
 	 Offset along an equidistant spiral with numPoints, out to a maximum radius of maxDither.

	 Adds columns: ['spiralDitherPerNightRa', 'spiralDitherPerNightDec']
- `ZenithDistStacker <lsst.sims.maf.stackers.html#lsst.sims.maf.stackers.generalStackers.ZenithDistStacker>`_ 
 	 Calculate the zenith distance for each pointing.

	 Adds columns: ['zenithDistance']
 
