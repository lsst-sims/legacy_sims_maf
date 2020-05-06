import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from lsst.sims.maf.utils.snNSNUtils import Load_Reference, Telescope, LCfast
from lsst.sims.maf.utils.snNSNUtils import SN_Rate, CovColor
import pandas as pd
import numpy.lib.recfunctions as rf
import time
from scipy.interpolate import interp1d


class SNNSNMetric(BaseMetric):
    """
    Estimate (nSN,zlim) of type Ia supernovae.

    Parameters
    ---------------
    metricName : str, opt
      metric name (default : SNSNRMetric)
    mjdCol : str, opt
      mjd column name (default : observationStartMJD)
    RACol : str,opt
      Right Ascension column name (default : fieldRA)
    DecCol : str,opt
      Declinaison column name (default : fieldDec)
    filterCol : str,opt
       filter column name (default: filter)
    m5Col : str, opt
       five-sigma depth column name (default : fiveSigmaDepth)
    exptimeCol : str,opt
       exposure time column name (default : visitExposureTime)
    nightCol : str,opt
       night column name (default : night)
    obsidCol : str,opt
      observation id column name (default : observationId)
    nexpCol : str,opt
      number of exposure column name (default : numExposures)
     vistimeCol : str,opt
        visit time column name (default : visitTime)
    season : list,opt
       list of seasons to process (float)(default: -1 = all seasons)
    zmin : float,opt
       min redshift for the study (default: 0.0)
    zmax : float,opt
       max redshift for the study (default: 1.2)
    pixArea: float, opt
       pixel area (default: 9.6)
    verbose: bool,opt
      verbose mode (default: False)
   ploteffi: bool,opt
      to plot observing efficiencies vs z (default: False)
    n_bef: int, opt
      number of LC points LC before T0 (default:5)
    n_aft: int, opt
      number of LC points after T0 (default: 10)
     snr_min: float, opt
       minimal SNR of LC points (default: 5.0)
     n_phase_min: int, opt
       number of LC points with phase<= -5(default:1)
    n_phase_max: int, opt
      number of LC points with phase>= 20 (default: 1)
    """

    def __init__(self, metricName='SNNSNMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', season=[-1], zmin=0.0, zmax=1.2,
                 pixArea=9.6, verbose=False, ploteffi=False,
                 n_bef=5, n_aft=10, snr_min=5., n_phase_min=1,
                 n_phase_max=1, templateDir='reference_files', **kwargs):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RACol = RACol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = 'season'
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol
        self.pixArea = pixArea

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol]

        super(SNNSNMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

        self.season = season

        # LC selection parameters
        self.n_bef = n_bef  # nb points before peak
        self.n_aft = n_aft  # nb points after peak
        self.snr_min = snr_min  # SNR cut for points before/after peak
        self.n_phase_min = n_phase_min  # nb of point with phase <=-5
        self.n_phase_max = n_phase_max  # nb of points with phase >=20

        # loading reference LC files
        lc_reference = Load_Reference(templateDir=templateDir).ref

        self.lcFast = {}
        telescope = Telescope(airmass=1.2)
        for key, vals in lc_reference.items():
            self.lcFast[key] = LCfast(vals, key[0], key[1], telescope,
                                      self.mjdCol, self.RACol, self.DecCol,
                                      self.filterCol, self.exptimeCol,
                                      self.m5Col, self.seasonCol,
                                      self.snr_min)

        # loading parameters
        self.zmin = zmin  # zmin for the study
        self.zmax = zmax  # zmax for the study
        self.zStep = 0.05  # zstep
        self.daymaxStep = 3.  # daymax step
        self.min_rf_phase = -20.  # min ref phase for LC points selection
        self.max_rf_phase = 40.  # max ref phase for LC points selection

        self.min_rf_phase_qual = -15.  # min ref phase for bounds effects
        self.max_rf_phase_qual = 25.  # max ref phase for bounds effects

        # snrate
        self.rateSN = SN_Rate(
            min_rf_phase=self.min_rf_phase_qual, max_rf_phase=self.max_rf_phase_qual)

        # verbose mode - useful for debug and code performance estimation
        self.verbose = verbose
        self.ploteffi = ploteffi

        # supernovae parameters
        self.params = ['x0', 'x1', 'daymax', 'color']

    def run(self, dataSlice,  slicePoint=None):
        """
        """
        if slicePoint is not None:
            self.pixArea = hp.nside2pixarea(slicePoint['nside'], degrees=True)

        # Two things to do: concatenate data (per band, night) and estimate seasons
        dataSlice = self.coadd(pd.DataFrame(dataSlice))

        dataSlice = self.getseason(dataSlice)

        # get the seasons
        seasons = self.season

        # if seasons = -1: process the seasons seen in data
        if self.season == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # get redshift range for processing
        zRange = list(np.arange(self.zmin, self.zmax, self.zStep))
        if zRange[0] < 1.e-6:
            zRange[0] = 0.01

        self.zRange = np.unique(zRange)
        # season infos
        dfa = pd.DataFrame(np.copy(dataSlice))
        season_info = dfa.groupby(['season']).apply(
            lambda x: self.seasonInfo(x)).reset_index()

        # select seasons of at least 30 days
        idx = season_info['season_length'] >= 30
        season_info = season_info[idx]

        # check wether requested seasons can be processed
        test_season = season_info[season_info['season'].isin(seasons)]
        # if len(test_season) == 0:
        if test_season.empty:
            return -1., -1.
        else:
            seasons = test_season['season']

        # get season length depending on the redshift
        dur_z = season_info.groupby(['season']).apply(
            lambda x: self.duration_z(x)).reset_index()

        # remove dur_z with negative season lengths
        idx = dur_z['season_length'] >= 10.
        dur_z = dur_z[idx]

        # generating simulation parameters
        gen_par = dur_z.groupby(['z', 'season']).apply(
            lambda x: self.calcDaymax(x)).reset_index()

        if gen_par.empty:
            return -1, -1

        resdf = pd.DataFrame()
        for seas in seasons:
            vara_df = self.run_season(
                dataSlice, [seas], gen_par, dur_z)
            resdf = pd.concat((resdf, vara_df))

        # final result: median zlim for a faint sn
        # and nsn_med for z<zlim

        idx = np.abs(resdf['x1']+2.0) < 1.e-5

        resdf = resdf.round({'zlim': 3, 'nsn_med': 3})
        zlim = resdf[idx]['zlim'].median()
        nSN = resdf[idx]['nsn_med'].sum()

        return nSN, zlim

    def coadd(self, data):
        """
        Method to coadd data per band and per night

        Parameters
        ---------------
        data: pandas df of observations

        Returns
        -----------
        coadded data (pandas df)

        """

        keygroup = [self.filterCol, self.nightCol]

        data.sort_values(by=keygroup, ascending=[True, True], inplace=True)

        coadd_df = data.groupby(keygroup).agg({self.nexpCol: ['sum'],
                                               self.vistimeCol: ['sum'],
                                               self.exptimeCol: ['sum'],
                                               self.mjdCol: ['mean'],
                                               self.RACol: ['mean'],
                                               self.DecCol: ['mean'],
                                               self.m5Col: ['mean']}).reset_index()

        coadd_df.columns = [self.filterCol, self.nightCol, self.nexpCol,
                            self.vistimeCol, self.exptimeCol, self.mjdCol,
                            self.RACol, self.DecCol, self.m5Col]

        coadd_df.loc[:, self.m5Col] += 1.25 * \
            np.log10(coadd_df[self.vistimeCol]/30.)

        coadd_df.sort_values(by=[self.filterCol, self.nightCol], ascending=[
                             True, True], inplace=True)

        return coadd_df.to_records(index=False)

    def getseason(self, obs, season_gap=80., mjdCol='observationStartMJD'):
        """
        Method to estimate seasons
        Parameters
        --------------
        obs: numpy array
          array of observations
        season_gap: float, opt
         minimal gap required to define a season (default: 80 days)
        mjdCol: str, opt
        col name for MJD infos (default: observationStartMJD)

        Returns
        ----------
        original numpy array with seasonnumber appended
        """

        # check wether season has already been estimated

        if 'season' in obs.dtype.names:
            return obs

        obs.sort(order=mjdCol)

        seasoncalc = np.ones(obs.size, dtype=int)

        if len(obs) > 1:
            diff = np.diff(obs[mjdCol])
            flag = np.where(diff > season_gap)[0]

            if len(flag) > 0:
                for i, indx in enumerate(flag):
                    seasoncalc[indx+1:] = i+2

        obs = rf.append_fields(obs, 'season', seasoncalc)

        return obs

    def seasonInfo(self, grp):
        """
        Method to estimate seasonal info (cadence, season length, ...)
        Parameters
        --------------
        grp: pandas df group
        Returns
        ---------
        pandas df with the cfollowing cols:
        """
        df = pd.DataFrame([len(grp)], columns=['Nvisits'])
        df['MJD_min'] = grp[self.mjdCol].min()
        df['MJD_max'] = grp[self.mjdCol].max()
        df['season_length'] = df['MJD_max']-df['MJD_min']
        df['cadence'] = 0.

        if len(grp) > 5:
            to = grp.groupby(['night'])[self.mjdCol].median().sort_values()
            df['cadence'] = np.mean(to.diff())

        return df

    def duration_z(self, grp):
        """
        Method to estimate the season length vs redshift
        This is necessary to take into account boundary effects
        when estimating the number of SN that can be detected
        daymin, daymax = min and max MJD of a season
        T0_min(z) =  daymin-(1+z)*min_rf_phase_qual
        T0_max(z) =  daymax-(1+z)*max_rf_phase_qual
        season_length(z) = T0_max(z)-T0_min(z)
        Parameters
        --------------
        grp: pandas df group
          data to process: season infos
        Returns
        ----------
        pandas df with season_length, z, T0_min and T0_max cols
        """

        daymin = grp['MJD_min'].values
        daymax = grp['MJD_max'].values
        dur_z = pd.DataFrame(self.zRange, columns=['z'])
        dur_z['T0_min'] = daymin-(1.+dur_z['z'])*self.min_rf_phase_qual
        dur_z['T0_max'] = daymax-(1.+dur_z['z'])*self.max_rf_phase_qual
        dur_z['season_length'] = dur_z['T0_max']-dur_z['T0_min']

        return dur_z

    def calcDaymax(self, grp):
        """
        Method to estimate T0 (daymax) values for simulation.
        Parameters
        --------------
        grp: group (pandas df sense)
         group of data to process with the following cols:
           T0_min: T0 min value (per season)
           T0_max: T0 max value (per season)
        Returns
        ----------
        pandas df with daymax, min_rf_phase, max_rf_phase values
        """

        T0_max = grp['T0_max'].values
        T0_min = grp['T0_min'].values
        num = (T0_max-T0_min)/self.daymaxStep
        if T0_max-T0_min > 10:
            df = pd.DataFrame(np.linspace(
                T0_min, T0_max, int(num)), columns=['daymax'])
        else:
            df = pd.DataFrame([-1], columns=['daymax'])

        df['min_rf_phase'] = self.min_rf_phase_qual
        df['max_rf_phase'] = self.max_rf_phase_qual

        return df

    def run_season(self, dataSlice, season, gen_par, dura_z):
        """
        Method to run on seasons
        Parameters
        --------------
        dataSlice: numpy array, opt
          data to process (scheduler simulations)
        seasons: list(int)
          list of seasons to process
        Returns
        ---------
        effi_seasondf: pandas df
          efficiency curves
        zlimsdf: pandas df
          redshift limits and number of supernovae
        """

        time_ref = time.time()

        if self.verbose:
            print('#### Processing season', season)

        groupnames = ['season', 'x1', 'color']

        gen_p = gen_par[gen_par['season'].isin(season)]

        if gen_p.empty:
            if self.verbose:
                print('No generator parameter found')
            return None
        dur_z = dura_z[dura_z['season'].isin(season)]
        obs = pd.DataFrame(np.copy(dataSlice))
        obs = obs[obs['season'].isin(season)]

        # simulate supernovae and lc
        if self.verbose:
            print("SN generation")
        sn = self.genSN(obs.to_records(
            index=False), gen_p.to_records(index=False))

        if self.verbose:
            print('sn and lc', len(sn), sn[['z', 'Cov_colorcolor']])

        # from these supernovae: estimate observation efficiency vs z
        effi_seasondf = self.effidf(sn)

        # zlims can only be estimated if efficiencies are ok
        idx = effi_seasondf['z'] <= 0.2
        x1ref = -2.0
        colorref = 0.2
        idx &= np.abs(effi_seasondf['x1']-x1ref) < 1.e-5
        idx &= np.abs(effi_seasondf['color']-colorref) < 1.e-5
        sel = effi_seasondf[idx]

        # zlims can only be estimated if efficiencies are high enough
        idx = effi_seasondf['z'] <= 0.2
        x1ref = -2.0
        colorref = 0.2
        idx &= np.abs(effi_seasondf['x1']-x1ref) < 1.e-5
        idx &= np.abs(effi_seasondf['color']-colorref) < 1.e-5
        sel = effi_seasondf[idx]

        if np.mean(sel['effi']) > 0.10:
                # estimate zlims
            zlimsdf = self.zlims(effi_seasondf, dur_z, groupnames)

            # estimate number of medium supernovae
            zlimsdf['nsn_med'],  zlimsdf['var_nsn_med'] = zlimsdf.apply(lambda x: self.nsn_typedf(
                x, 0.0, 0.0, effi_seasondf, dur_z), axis=1, result_type='expand').T.values

        if self.verbose:
            print('#### SEASON processed', time.time()-time_ref,
                  season)

        return zlimsdf

    def genSN(self, obs, gen_par):
        """
        Method to simulate LC and supernovae
        Parameters
        ---------------
        obs: numpy array
          array of observations(from scheduler)
        gen_par: numpy array
          array of parameters for simulation
        """

        time_ref = time.time()
        # LC estimation

        sn_tot = pd.DataFrame()
        lc_tot = pd.DataFrame()
        for key, vals in self.lcFast.items():
            time_refs = time.time()
            gen_par_cp = np.copy(gen_par)
            if key == (-2.0, 0.2):
                idx = gen_par_cp['z'] < 0.9
                gen_par_cp = gen_par_cp[idx]
            lc = vals(obs, gen_par_cp, bands='grizy')
            if self.verbose:
                print('End of simulation', key, time.time()-time_refs)

            if self.verbose:
                print('End of simulation after concat',
                      key, time.time()-time_refs)

            # estimate SN

            sn = pd.DataFrame()
            if len(lc) > 0:
                sn = self.process(pd.DataFrame(lc))

            if self.verbose:
                print('End of supernova', time.time()-time_refs)

            if not sn.empty:
                sn_tot = pd.concat([sn_tot, pd.DataFrame(sn)], sort=False)

        if self.verbose:
            print('End of supernova - all', time.time()-time_ref)

        return sn_tot

    def process(self, tab):
        """
        Method to process LC: sigma_color estimation and LC selection
        Parameters
        --------------
        tab: pandas df of LC points with the following cols:
          flux:  flux
          fluxerr: flux error
          phase:  phase
          snr_m5: Signal-to-Noise Ratio
          time: time(MJD)
          mag: magnitude
          m5:  five-sigma depth
          magerr: magnitude error
          exposuretime: exposure time
          band: filter
          zp:  zero-point
          season: season number
          healpixID: pixel ID
          pixRA: pixel RA
          pixDec: pixel Dec
          z: redshift
          daymax: T0
          flux_e_sec: flux(in photoelec/sec)
          flux_5: 5-sigma flux(in photoelec/sec)
          F_x0x0, ...F_colorcolor: Fisher matrix elements
          x1: x1 SN
          color: color SN
          n_aft: number of LC points before daymax
          n_bef: number of LC points after daymax
          n_phmin: number of LC points with a phase < -5
          n_phmax:  number of LC points with a phase > 20
        Returns
        ----------
        """
        # now groupby
        tab = tab.round({'daymax': 3,
                         'z': 3, 'x1': 2, 'color': 2})
        groups = tab.groupby(
            ['daymax', 'season', 'z', 'x1', 'color'])

        tosum = []
        for ia, vala in enumerate(self.params):
            for jb, valb in enumerate(self.params):
                if jb >= ia:
                    tosum.append('F_'+vala+valb)
        tosum += ['n_aft', 'n_bef', 'n_phmin', 'n_phmax']
        # apply the sum on the group
        sums = groups[tosum].sum().reset_index()

        # select LC according to the number of points bef/aft peak
        idx = sums['n_aft'] >= self.n_aft
        idx &= sums['n_bef'] >= self.n_bef
        idx &= sums['n_phmin'] >= self.n_phase_min
        idx &= sums['n_phmax'] >= self.n_phase_max

        finalsn = pd.DataFrame()
        goodsn = pd.DataFrame(sums.loc[idx])

        # estimate the color for SN that passed the selection cuts
        if len(goodsn) > 0:
            goodsn.loc[:, 'Cov_colorcolor'] = CovColor(goodsn).Cov_colorcolor
            finalsn = pd.concat([finalsn, goodsn], sort=False)

        badsn = pd.DataFrame(sums.loc[~idx])

        # Supernovae that did not pass the cut have a sigma_color=10
        if len(badsn) > 0:
            badsn.loc[:, 'Cov_colorcolor'] = 100.
            finalsn = pd.concat([finalsn, badsn], sort=False)

        return finalsn

    def effidf(self, sn_tot, color_cut=0.04):
        """
        Method estimating efficiency vs z for a sigma_color cut
        Parameters
        ---------------
        sn_tot: pandas df
          data used to estimate efficiencies
        color_cut: float, opt
          color selection cut(default: 0.04)
        Returns
        ----------
        effi: pandas df with the following cols:
          season: season
          pixRA: RA of the pixel
          pixDec: Dec of the pixel
          healpixID: pixel ID
          x1: SN stretch
          color: SN color
          z: redshift
          effi: efficiency
          effi_err: efficiency error(binomial)
        """

        sndf = pd.DataFrame(sn_tot)

        listNames = ['season', 'x1', 'color']
        groups = sndf.groupby(listNames)

        # estimating efficiencies
        effi = groups[['Cov_colorcolor', 'z']].apply(
            lambda x: self.effiObsdf(x, color_cut)).reset_index(level=list(range(len(listNames))))

        # this is to plot efficiencies and also sigma_color vs z
        if self.ploteffi:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots()
            figb, axb = plt.subplots()

            self.plot(ax, effi, 'effi', 'effi_err',
                      'Observing Efficiencies', ls='-')
            sndf['sigma_color'] = np.sqrt(sndf['Cov_colorcolor'])
            self.plot(axb, sndf, 'sigma_color', None, '$\sigma_{color}$')
            # get efficiencies vs z

            plt.show()

        return effi

    def plot(self, ax, effi, vary, erry=None, legy='', ls='None'):
        """
        Simple method to plot vs z
        Parameters
        --------------
        ax:
          axis where to plot
        effi: pandas df
          data to plot
        vary: str
          variable(column of effi) to plot
        erry: str, opt
          error on y-axis(default: None)
        legy: str, opt
          y-axis legend(default: '')
        """
        grb = effi.groupby(['x1', 'color'])
        yerr = None
        for key, grp in grb:
            x1 = grp['x1'].unique()[0]
            color = grp['color'].unique()[0]
            if erry is not None:
                yerr = grp[erry]
            ax.errorbar(grp['z'], grp[vary], yerr=yerr,
                        marker='o', label='(x1,color)=({},{})'.format(x1, color), lineStyle=ls)

        ftsize = 15
        ax.set_xlabel('z', fontsize=ftsize)
        ax.set_ylabel(legy, fontsize=ftsize)
        ax.xaxis.set_tick_params(labelsize=ftsize)
        ax.yaxis.set_tick_params(labelsize=ftsize)
        ax.legend(fontsize=ftsize)

    def effiObsdf(self, data, color_cut=0.04):
        """
        Method to estimate observing efficiencies for supernovae
        Parameters
        --------------
        data: pandas df - grp
          data to process
        Returns
        ----------
        pandas df with the following cols:
          - cols used to make the group
          - effi, effi_err: observing efficiency and associated error
        """

        # reference df to estimate efficiencies
        df = data.loc[lambda dfa:  np.sqrt(dfa['Cov_colorcolor']) < 100000., :]

        # selection on sigma_c<= 0.04
        df_sel = df.loc[lambda dfa:  np.sqrt(
            dfa['Cov_colorcolor']) <= color_cut, :]

        # make groups (with z)
        group = df.groupby('z')
        group_sel = df_sel.groupby('z')

        # Take the ratio to get efficiencies
        rb = (group_sel.size()/group.size())
        err = np.sqrt(rb*(1.-rb)/group.size())
        var = rb*(1.-rb)*group.size()

        rb = rb.array
        err = err.array
        var = var.array

        rb[np.isnan(rb)] = 0.
        err[np.isnan(err)] = 0.
        var[np.isnan(var)] = 0.

        return pd.DataFrame({group.keys: list(group.groups.keys()),
                             'effi': rb,
                             'effi_err': err,
                             'effi_var': var})

    def zlims(self, effi_seasondf, dur_z, groupnames):
        """
        Method to estimate redshift limits
        Parameters
        --------------
        effi_seasondf: pandas df
            season: season
          pixRA: RA of the pixel
          pixDec: Dec of the pixel
          healpixID: pixel ID
          x1: SN stretch
          color: SN color
          z: redshift
          effi: efficiency
          effi_err: efficiency error (binomial)
        dur_z: pandas df with the following cols:
           season: season
           z: redshift
           T0_min: min daymax
           T0_max: max daymax
            season_length: season length
        groupnames: list(str)
          list of columns to use to define the groups
        Returns
        ----------
        pandas df with the following cols: pixRA: RA of the pixel
           pixDec: Dec of the pixel
           healpixID: pixel ID
           season: season number
           x1: SN stretch
           color: SN color
           zlim: redshift limit
        """

        res = effi_seasondf.groupby(groupnames).apply(
            lambda x: self.zlimdf(x, dur_z)).reset_index(level=list(range(len(groupnames))))

        return res

    def zlimdf(self, grp, duration_z):
        """
        Method to estimate redshift limits
        Parameters
        --------------
        grp: pandas df group
          efficiencies to estimate redshift limits;
          columns:
           season: season
           pixRA: RA of the pixel
           pixDec: Dec of the pixel
           healpixID: pixel ID
           x1: SN stretch
           color: SN color
           z: redshift
           effi: efficiency
           effi_err: efficiency error (binomial)
        duration_z: pandas df with the following cols:
           season: season
           z: redshift
           T0_min: min daymax
           T0_max: max daymax
            season_length: season length
        Returns
        ----------
        pandas df with the following cols:
         zlimit: redshift limit
        """

        zlimit = 0.0

        # z range for the study
        zplot = list(np.arange(self.zmin, self.zmax, 0.001))

        # print(grp['z'], grp['effi'])

        if len(grp['z']) <= 3:
            return pd.DataFrame({'zlim': [zlimit]})
            # 'status': [int(status)]})
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            grp['z'], grp['effi'], kind='linear', bounds_error=False, fill_value=0.)

        # get rate
        season = np.median(grp['season'])
        idx = duration_z['season'] == season
        seas_duration_z = duration_z[idx]

        durinterp_z = interp1d(
            seas_duration_z['z'], seas_duration_z['season_length'], bounds_error=False, fill_value=0.)

        # estimate the rates and nsn vs z
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=self.zmin,
                                                       zmax=self.zmax,
                                                       duration_z=durinterp_z,
                                                       survey_area=self.pixArea)

        # rate interpolation
        rateInterp = interp1d(zz, nsn, kind='linear',
                              bounds_error=False, fill_value=0)

        # estimate the cumulated number of SN vs z
        nsn_cum = np.cumsum(effiInterp(zplot)*rateInterp(zplot))

        if nsn_cum[-1] >= 1.e-5:
            nsn_cum_norm = nsn_cum/nsn_cum[-1]  # normalize
            zlim = interp1d(nsn_cum_norm, zplot)
            zlimit = zlim(0.95).item()
            # status = self.status['ok']

            if self.ploteffi:
                import matplotlib.pylab as plt
                fig, ax = plt.subplots()
                x1 = grp['x1'].unique()[0]
                color = grp['color'].unique()[0]

                ax.plot(zplot, nsn_cum_norm,
                        label='(x1,color)=({},{})'.format(x1, color))

                ftsize = 15
                ax.set_ylabel('NSN ($z<$)', fontsize=ftsize)
                ax.set_xlabel('z', fontsize=ftsize)
                ax.xaxis.set_tick_params(labelsize=ftsize)
                ax.yaxis.set_tick_params(labelsize=ftsize)
                ax.set_xlim((0.0, 1.2))
                ax.set_ylim((0.0, 1.05))
                ax.plot([0., 1.2], [0.95, 0.95], ls='--', color='k')
                plt.legend(fontsize=ftsize)
                plt.show()

        return pd.DataFrame({'zlim': [zlimit]})
        # 'status': [int(status)]})

    def nsn_typedf(self, grp, x1, color, effi_tot, duration_z, search=True):
        """
        Method to estimate the number of supernovae for a given type of SN
        Parameters
        --------------
        grp: pandas series with the following infos:
         pixRA: pixelRA
         pixDec: pixel Dec
         healpixID: pixel ID
         season: season
         x1: SN stretch
         color: SN color
         zlim: redshift limit
        x1, color: SN params to estimate the number
        effi_tot: pandas df with columns:
           season: season
           pixRA: RA of the pixel
           pixDec: Dec of the pixel
           healpixID: pixel ID
           x1: SN stretch
           color: SN color
           z: redshift
           effi: efficiency
           effi_err: efficiency error (binomial)
        duration_z: pandas df with the following cols:
           season: season
           z: redshift
           T0_min: min daymax
           T0_max: max daymax
            season_length: season length
        Returns
        ----------
        nsn: float
           number of supernovae
        """

        # get rate
        season = np.median(grp['season'])
        idx = duration_z['season'] == season
        seas_duration_z = duration_z[idx]

        durinterp_z = interp1d(
            seas_duration_z['z'], seas_duration_z['season_length'], bounds_error=False, fill_value=0.)

        if search:
            effisel = effi_tot.loc[lambda dfa: (
                dfa['x1'] == x1) & (dfa['color'] == color), :]
        else:
            effisel = effi_tot

        nsn, var_nsn = self.nsn(effisel, grp['zlim'], durinterp_z)

        return (nsn, var_nsn)

    def nsn(self, effi, zlim, duration_z):
        """
        Method to estimate the number of supernovae
        Parameters
        --------------
        effi: pandas df grp of efficiencies
          season: season
          pixRA: RA of the pixel
          pixDec: Dec of the pixel
          healpixID: pixel ID
          x1: SN stretch
          color: SN color
          z: redshift
          effi: efficiency
          effi_err: efficiency error (binomial)
        zlim: float
          redshift limit value
        duration_z: pandas df with the following cols:
           season: season
           z: redshift
           T0_min: min daymax
           T0_max: max daymax
            season_length: season length
        Returns
        ----------
        nsn, var_nsn : float
          number of supernovae (and variance) with z<zlim
        """

        if zlim < 1.e-3:
            return -1.0, -1.0

        dz = 0.001
        zplot = list(np.arange(self.zmin, self.zmax, dz))
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            effi['z'], effi['effi'], kind='linear', bounds_error=False, fill_value=0.)
        # estimate the cumulated number of SN vs z
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=self.zmin,
                                                       zmax=self.zmax,
                                                       dz=dz,
                                                       duration_z=duration_z,
                                                       survey_area=self.pixArea)

        nsn_cum = np.cumsum(effiInterp(zplot)*nsn)

        nsn_interp = interp1d(zplot, nsn_cum)

        nsn = nsn_interp(zlim).item()

        return [nsn, 0.0]
