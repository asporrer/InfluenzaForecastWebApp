$("#confirmMultiselectWaveStatistics").mouseup(function(){
    var text = $("#waveStatisticsMultiselect").val();

    $.ajax({
      url: "/waveStatisticsFigure",
      type: "get",
      data: {jsdata: text},
      success: function(response) {
        $("#placeForWaveStatisticsFigure").html(response);
      },
      error: function(xhr) {
        //Do Something to handle error
      }
    });
});

$("#confirmMultiselectWaveStartVsInensityStatistics").mouseup(function(){
    var text = $("#waveStartVsIntensityMultiselect").val();

    $.ajax({
      url: "/waveStartVsIntensityFigure",
      type: "get",
      data: {jsdata: text},
      success: function(response) {
        $("#placeForWaveStartVsIntensityFigure").html(response);
      },
      error: function(xhr) {
        //Do Something to handle error
      }
    });
});

$("#featuresSingleSelect").change(function(){
    var text = $("#featuresSingleSelect").val();

    $.ajax({
      url: "/featuresFigure",
      type: "get",
      data: {jsdata: text},
      success: function(response) {
        $("#placeForFeaturesFigure").html(response);
      },
      error: function(xhr) {
        //Do Something to handle error
      }
    });
});