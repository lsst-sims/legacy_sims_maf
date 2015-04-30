(function(){
    var app = angular.module('searchMetrics', ['smart-table']);
    app.config(function ($interpolateProvider) {
        $interpolateProvider.startSymbol('{$');
        $interpolateProvider.endSymbol('$}');
    });

    app.controller('searchController', ['$scope', '$http', '$timeout', function($scope, $http, $timeout){
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
                $("#loading").hide();
            });

        };
        load_all_metrics();

        var search_mode = false;

        $.get('/search?list_type=sim_data')
          .done(function(data){
            $scope.sim_data = JSON.parse(data);
            $scope.$digest();
        });

        $.get('/search?list_type=slicer')
          .done(function(data){
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
