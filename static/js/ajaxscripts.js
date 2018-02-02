$("#confirmMultiselectWaveStatistics").mouseup(function(){
    var text2 = $("#waveStatisticsMultiselect").val();

    $.ajax({
      url: "/waveStatisticsFigure",
      type: "get",
      data: {jsdata: text2},
      success: function(response) {
        $("#placeForWaveStatisticsFigure").html(response);
      },
      error: function(xhr) {
        //Do Something to handle error
      }
    });
});
