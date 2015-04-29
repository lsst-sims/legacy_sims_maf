(function(){
    var app = angular.module('searchMetrics', ['smart-table']);
    app.config(function ($interpolateProvider) {
        $interpolateProvider.startSymbol('{$');
        $interpolateProvider.endSymbol('$}');
    });

    app.controller('searchController', ['$scope', '$http', '$timeout', function($scope, $http, $timeout){
        //$scope.runs = [{'opsimComment': '10 yr, 10 cheeses w/undercheese', 'mafComment': 'maf_cadence_ops1_1122', 'opsimDate': '05/18/14', 'mafDir': 'maf_cadence/maf_cadence_ops1_1122', 'mafDate': '11/11/14', 'mafRunId': 1, 'opsimRun': 'ops1_1122'}, {'opsimComment': 'tentative baseline min Alt=20', 'mafComment': 'maf_cadence_ops1_1140', 'opsimDate': '07/29/14', 'mafDir': 'maf_cadence/maf_cadence_ops1_1140', 'mafDate': '11/10/14', 'mafRunId': 2, 'opsimRun': 'ops1_1140'}, {'opsimComment': 'tier 1 #5 min Alt=20', 'mafComment': 'maf_cadence_ops1_1141', 'opsimDate': '07/29/14', 'mafDir': 'maf_cadence/maf_cadence_ops1_1141', 'mafDate': '11/10/14', 'mafRunId': 3, 'opsimRun': 'ops1_1141'}, {'opsimComment': 'tier 1 #9 min Alt=20', 'mafComment': 'maf_cadence_ops1_1144', 'opsimDate': '07/29/14', 'mafDir': 'maf_cadence/maf_cadence_ops1_1144', 'mafDate': '11/11/14', 'mafRunId': 4, 'opsimRun': 'ops1_1144'}, {'opsimComment': 'tier 1 #10 min Alt=20', 'mafComment': 'maf_cadence_ops1_1146', 'opsimDate': '07/29/14', 'mafDir': 'maf_cadence/maf_cadence_ops1_1146', 'mafDate': '11/11/14', 'mafRunId': 5, 'opsimRun': 'ops1_1146'}, {'opsimComment': 'tier 1 #6 min Alt=20', 'mafComment': 'maf_cadence_ops1_1147', 'opsimDate': '07/29/14', 'mafDir': 'maf_cadence/maf_cadence_ops1_1147', 'mafDate': '11/10/14', 'mafRunId': 6, 'opsimRun': 'ops1_1147'}];
        $scope.metric_list = [];
        $scope.metrics = [];
        $scope.itemsByPage = 10;
        $scope.show_data = false;

        var load_all_metrics = function(){
            $.get('/search?list_type=metrics')
              .done(function(data){
                // $('#results').text(data);
                data = JSON.parse(data);
                $scope.metric_list.push.apply($scope.metric_list, data);
                $scope.metrics = [].concat(data);
                $scope.$digest();
            });

        };
        load_all_metrics();

        var search_mode = false;

        $.get('/search?list_type=sim_data')
          .done(function(data){
            // $('#results').text(data);
            $scope.sim_data = JSON.parse(data);
            $scope.$digest();
        });

        $.get('/search?list_type=slicer')
          .done(function(data){
            // $('#results').text(data);
            $scope.slicer = JSON.parse(data);
            $scope.$digest();
        });

        $scope.search = function(){
            var keywords = {};
            if ( $('#name').val() != "" ){
                var names = $('#name').val().split(',');
                names.forEach(function(d){ d.trim(); });
                keywords.name = names;
            }
            var sim_data = [];
            $('#sim_data input[type=checkbox]:checked').each(function(e){ return sim_data.push(this.value); });
            if ( sim_data.length > 0 ){
                keywords.sim_data = sim_data;
            }

            var slicer = [];
            $('#slicer input[type=checkbox]:checked').each(function(e){ return slicer.push(this.value); });
            if ( slicer.length > 0 ){
                keywords.slicer = slicer;
            }
            console.log(keywords);
            $.post('/search', {"keywords": JSON.stringify(keywords)})
              .done(function(data){
                // $('#results').text(data);
                data = JSON.parse(data);
                $scope.metric_list.push.apply($scope.metric_list, data);
                $scope.metrics = [].concat(data);
                search_mode = true;
                $scope.$digest();
            });

        };
        $scope.reset = function(){
            if (search_mode == true){
                load_all_metrics();
                search_mode = false;
                $timeout(function(){
                    $('#filter-reset').click();
                }, 100);
            }

        };

        var get_plot_name = function(src){
            return src.replace('.pdf', '');
        };

        $scope.get_plots = function(metric){
            var plots = [];
            if ( typeof(metric.plots.SkyMap) !== "undefined" ){
                plots.push(get_plot_name(metric.plots.SkyMap));
            }
            if ( typeof(metric.plots.Histogram) !== "undefined" ){
                plots.push(get_plot_name(metric.plots.Histogram));
            }
            for ( var key in metric.plots ){
               if ( key != "SkyMap" && key != "Histogram" ){
                   if ( typeof(metric.plots[key]) === "undefined"){
                       continue;
                   }
                   plots.push(get_plot_name(metric.plots[key]));
               }
            }
            return plots;
        };

        $scope.get_thumb_path = function(metric, plot){
            return metric.mafDir + '/' + 'thumb.' + plot + '.png';
        };
        $scope.get_pdf_path = function(metric, plot){
            return metric.mafDir + '/' + plot + '.pdf';
        };


    }]);

})();