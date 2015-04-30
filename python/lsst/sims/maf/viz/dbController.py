from lsst.sims.maf.db import trackingDb, resultsDb
import os
import json

class MetricObj(object):
    """
    Save a metric as an object
    """
    def __init__(self, metadata):
        self.metadata = metadata
        self.plots = {}
        self.stats = []

    def __repr__(self):
        return json.dumps(self.metadata)

    def __str__(self):
        return json.dumps(self.metadata)

    def to_json(self):
        return json.dumps(self.metadata)

    def info(self):
        results = self.metadata.copy()
        if len(self.plots.keys()) > 0:
            results['plots'] = self.plots
        if len(self.stats) > 0:
            results['stats'] = self.stats
        return results

class RunObj(object):
    """
    Save a run as an object
    """
    def __init__(self, metadata):
        self.metadata = metadata
        self.metrics = None
        self.run_db = resultsDb.ResultsDb(resultsDbAddress='sqlite:///' + self.metadata['mafDir'] + '/resultsDb_sqlite.db')
        # initialize dictionary
        self.metric_objs = {}
        self.load_metric_objs()

    def load_metric_objs(self):

        metadata_list = ['metricId', 'metricName', 'slicerName', 'metricMetadata', 'simDataName', 'metricDataFile',
                        'displayGroup', 'displaySubgroup', 'displayOrder', 'displayCaption']

        # join metrics and displays
        sql = 'SELECT A.metricId, ' + ', '.join(metadata_list[1:]) + ' FROM displays AS A, metrics AS B WHERE A.metricId = B.metricId'
        metrics = self.run_db.session.execute(sql).fetchall()

        for metric in metrics:
            metadata = {}
            for m in metadata_list:
                metadata[m] = getattr(metric, m)
            metric_obj = MetricObj(metadata)
            self.metric_objs[metadata['metricId']] = metric_obj
            metric_obj.run = self
            metric_obj.metadata['mafRunId'] = self.metadata['mafRunId']
            metric_obj.metadata['mafDir'] = os.path.relpath(self.metadata['mafDir'], os.getcwd()) 

        # get all plots
        plots = self.run_db.session.query(resultsDb.PlotRow).all()
        for plot in plots:
            self.metric_objs[plot.metricId].plots[plot.plotType] = plot.plotFile

        # get all stats
        stats = self.run_db.session.query(resultsDb.SummaryStatRow).all()
        for stat in stats:
            self.metric_objs[stat.metricId].stats.append({'summaryName': stat.summaryName,
                                                      'summaryValue': stat.summaryValue})
    def __repr__(self):
        return json.dumps(self.metadata)


class ShowMafDBController(object):

    def __init__(self, tracking_db_address):
        self.tracking_db = trackingDb.TrackingDb(trackingDbAddress=tracking_db_address)
        self.run_objs = []
        self.all_metrics_idx = {}
        self.load_run_objs()
        self.build_metric_index()

    def load_run_objs(self):
        self.run_objs = []
        runs = self.tracking_db.session.query(trackingDb.RunRow).all()
        metadata_list = ['mafRunId', 'opsimRun', 'opsimComment', 'mafComment', 'mafDir', 'opsimDate', 'mafDate']
        for run in runs:
            metadata = {}
            for m in metadata_list:
                metadata[m] = getattr(run, m)
            run_obj = RunObj(metadata)
            self.run_objs.append(run_obj)

    def build_metric_index(self):
        """
        Building hash table index for searching.
        The metrics will be stored in the list at the corresponding bucket.
        """

        self.all_metrics = []
        self.all_metrics_idx['name'] = {}
        self.all_metrics_idx['sim_data'] = {}
        self.all_metrics_idx['slicer'] = {}

        for run_obj in self.run_objs:
            for idx in run_obj.metric_objs:
                metric_obj = run_obj.metric_objs[idx]
                self.all_metrics.append(metric_obj)

                # if the index not exist, init the list
                if metric_obj.metadata['metricName'] not in self.all_metrics_idx['name']:
                    self.all_metrics_idx['name'][metric_obj.metadata['metricName']] = []

                self.all_metrics_idx['name'][metric_obj.metadata['metricName']].append(metric_obj)

                # if the index not exist, init the list
                if metric_obj.metadata['simDataName'] not in self.all_metrics_idx['sim_data']:
                    self.all_metrics_idx['sim_data'][metric_obj.metadata['simDataName']] = []

                self.all_metrics_idx['sim_data'][metric_obj.metadata['simDataName']].append(metric_obj)

                # if the index not exist, init the list
                if metric_obj.metadata['slicerName'] not in self.all_metrics_idx['slicer']:
                    self.all_metrics_idx['slicer'][metric_obj.metadata['slicerName']] = []

                self.all_metrics_idx['slicer'][metric_obj.metadata['slicerName']].append(metric_obj)

    def get_all_metrics(self):
        return map(lambda x: x.info(), self.all_metrics)
    def get_all_sim_data(self):
        return self.all_metrics_idx['sim_data'].keys()
    def get_all_slicer(self):
        return self.all_metrics_idx['slicer'].keys()


    def search_metrics(self, keywords):
        """
        given search keywords, return a list of metrics
        :param keywords:

        {
            'name': ['metric_name_key_1', 'metric_name_key_2' ... ],
            'sim_data': 'sim_data_name',
            'slicer': 'slicer_name'
        }
        :return:
        """
        results = None

        if keywords.get('name'):

            # if this is the first seach category, initialize search results
            if results is None:
                results = []

            # iterate through all the name index to find match metric name
            for search_key in keywords.get('name'):
                for name_idx in self.all_metrics_idx['name']:
                    if name_idx.find(search_key) >= 0:
                        results.extend(self.all_metrics_idx['name'][name_idx])


        if keywords.get('sim_data'):



            # if this is the first seach category, initialize search results
            if results is None:
                results = []
                for search_key in keywords.get('sim_data'):
                    if search_key in self.all_metrics_idx['sim_data']:
                        results.extend(self.all_metrics_idx['sim_data'][search_key])

            else:
                new_results = []
                for metric in results:
                    for search_key in keywords.get('sim_data'):
                        if search_key == metric.metadata['simDataName']:
                            new_results.append(metric)
                results = new_results


        if keywords.get('slicer'):



            # if this is the first seach category, initialize search results
            if results is None:
                results = []
                for search_key in keywords.get('slicer'):
                    if search_key in self.all_metrics_idx['slicer']:
                        results.extend(self.all_metrics_idx['slicer'][search_key])

            else:
                new_results = []
                for metric in results:
                    for search_key in keywords.get('slicer'):
                        if search_key == metric.metadata['slicerName']:
                            new_results.append(metric)
                results = new_results


        return map(lambda x: x.info(), results)
