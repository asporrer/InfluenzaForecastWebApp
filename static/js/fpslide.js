var margin = {top:80, right:50, bottom:0, left:50};
var width = 1000 - margin.left - margin.right;
var height = 550 - margin.top - margin.bottom;

var svgWidth = width + margin.left + margin.right;
var svgHeight = height + margin.top + margin.bottom;


var coordinateSystemYBottom = 6/7 * svgHeight; // Distance to svg anchor
var coordinateSystemYTop = margin.top + 30;

var coordinateSystemXLeft = margin.left;
var coordinateSystemXRight = width + 30;


var topMarginxAxis = coordinateSystemYBottom;
var leftMarginyAxis = coordinateSystemXLeft;

var xScaleRangeStart = coordinateSystemXLeft;
var xScaleRangeEnd = coordinateSystemXRight;

var yScaleRangeStart = coordinateSystemYBottom;
var yScaleRangeEnd = coordinateSystemYTop;

var slider_range_start = margin.left; // 260 for layout with integrated buttons.
var slider_range_end = width + 30;

var ysliderRange = 30;


var actualValuesColor = "#696969";
var pred1Color = "#ff661a";
var pred2Color = "#ff3333";

var playButton = d3.select("#play-button");


// Drop Down Selection Menu:
var options = ["Bayern", "Bayern", "Bayern"];
var stateString = options[0];

var select = d3.select("#fpsDD")
    .on('change',onchange);

select
  .selectAll('option')
	.data(options).enter()
	.append('option')
		.text(function (d) { return d; });

function onchange() {
	selectValue = d3.select('select').property('value');
	updatePlotBasedOnSelection(selectValue);
};


var svg = d3.select("#svgId")
    .append("svg")
    .attr("width", svgWidth)
    .attr("height", svgHeight)
    .attr("viewBox", "0 0 1000 550")  // added
    .attr("preserveAspectRatio", "xMinYMin meet"); // added



// Axis Labels:
var axisLabelTextSize = 15;

svg.append("text")
   .text("Date")
   .attr("text-anchor", "middle")
   .attr("x", Math.round((xScaleRangeEnd - xScaleRangeStart)/2 + xScaleRangeStart) )
   .attr("y", topMarginxAxis + 47)
   .attr("font-family", "sans-serif")
   .attr("font-size", axisLabelTextSize)
   .attr("fill", "black");

svg.append("text")
   .text("# Reported Influenza Infections")
   .attr("text-anchor", "middle")
   .attr("font-family", "sans-serif")
   .attr("font-size", axisLabelTextSize)
   .attr("fill", "black")
    .attr("transform", "translate(" + (leftMarginyAxis - 30) + "," + Math.round((yScaleRangeEnd - yScaleRangeStart)/2 + yScaleRangeStart) + ")rotate(-90)");

// Legend
var xlegendLineStart = svgWidth * 7/9 - 25;
var ylegendLineStart = 2/9*svgHeight;

var ylegendDistance = 20;

var legendLineWidth = 20;
var legendLineHeight = 2;

var textSize = 12;
var xLegendTextStart = xlegendLineStart + legendLineWidth + 5;
var yLegendTextStart = ylegendLineStart + Math.round(textSize/2);

svg.append("rect")
    .attr("x", xlegendLineStart)
   .attr("y", ylegendLineStart)
   .attr("width", legendLineWidth)
   .attr("height", legendLineHeight)
   .attr("fill", actualValuesColor);

svg.append("text")
   .text("Actual Influenza Infections")
   .attr("text-anchor", "left")
   .attr("x", xLegendTextStart)
   .attr("y", yLegendTextStart)
   .attr("font-family", "sans-serif")
   .attr("font-size", textSize)
   .attr("fill", actualValuesColor);

svg.append("rect")
    .attr("x", xlegendLineStart)
   .attr("y", ylegendLineStart + ylegendDistance)
   .attr("width", legendLineWidth)
   .attr("height", legendLineHeight)
   .attr("fill", pred1Color);

svg.append("text")
   .text("Prediction: 0.8 < #Infections")
   .attr("text-anchor", "left")
   .attr("x", xLegendTextStart)
   .attr("y", yLegendTextStart + ylegendDistance)
   .attr("font-family", "sans-serif")
   .attr("font-size", textSize)
   .attr("fill", pred1Color);

svg.append("rect")
    .attr("x", xlegendLineStart)
   .attr("y", ylegendLineStart + 2*ylegendDistance)
   .attr("width", legendLineWidth)
   .attr("height", legendLineHeight)
   .attr("fill", pred2Color);

svg.append("text")
   .text("Prediction: 7.0 < #Infections")
   .attr("text-anchor", "left")
   .attr("x", xLegendTextStart)
   .attr("y", yLegendTextStart + 2*ylegendDistance)
   .attr("font-family", "sans-serif")
   .attr("font-size", textSize)
   .attr("fill", pred2Color);


var lineGraphActual;
var lineGraphPred1;
var lineGraphPred2;

var moving = false;
var currentIndex = 0;


var formatDateIntoMonthDayYear = d3.timeFormat("%d. %b %Y");
var formatDate = d3.timeFormat("%d %b");
var parseDate = d3.timeParse("%Y-%m-%d");


var startDate = parseDate("2000-01-01"),
    endDate = parseDate("2000-01-02");


var sliderDatesScale = d3.scaleTime()
    .domain([startDate, endDate])
    .range([slider_range_start, slider_range_end]);

var xScale = d3.scaleTime()
                .domain([startDate, endDate])
                .range([xScaleRangeStart, xScaleRangeEnd]);

var yScale = d3.scaleLinear()
            .domain([0, 60]) // Dynamic?
            .range([yScaleRangeStart, yScaleRangeEnd]);


var datelineArray;

var actualDatesInfluArrays;
var pred1DatesInfluArrays;
var pred2DatesInfluArrays;

var actualDatesInfluArray;
var pred1DatesInfluArray;
var pred2DatesInfluArray;


var currentDateHorizonArrays;
var futureDatesArrays;
var currentHorizonInfluenzaArrays;
var prediction1InfluArrays;
var prediction2InfluArrays;


////////// slider //////////

var slider = svg.append("g")
    .attr("class", "slider")
    .attr("transform", "translate(" + 0 + "," + ysliderRange  + ")");

// Line function for the step plot
var lineFunction = d3.line()
    .x(function (d) {
        return  xScale(d[0]); })
    .y(function (d) {
        return yScale(d[1]); })
    .curve(d3.curveStepAfter);

// Get the current index for the current value of the slider button.
function getCurrentIndex(curValParam) {
    var helperDatelineArray = datelineArray.filter(function(d) {
       return d <= sliderDatesScale.invert(curValParam)
    });

    if (helperDatelineArray.length == 0) {
        return 0; }
    else {
        return helperDatelineArray.length - 1;
    }
};


// Functions For Fata Preparation

function prepareWholeDateline(d) {
    return parseDate(d[0]);
};

function prepareDates(d) {
    var return_array = [];
    for (var prop in d) {
        if (d.hasOwnProperty(prop)) {
	       return_array.push(parseDate(d[prop]));
        };
    };
    return return_array;
};

function prepareValues(d) {
    var return_array = [];
    for (var prop in d) {
        if (d.hasOwnProperty(prop)) {
	       return_array.push(parseFloat(d[prop]));
        };
    };
    return return_array;
};

function getZippedDatasets(dataset1Par, dataset2Par) {
    var returnDataset = [];
    for (var i = 0; i < dataset1Par.length; i++) {
        returnDataset.push([]);
        for (var j = 0; j < dataset1Par.columns.length; j++) {
            returnDataset[i].push([dataset1Par[i][j], dataset2Par[i][j]]);
        }
    }
    return returnDataset;
};


d3.csv("/static/data/v1/wholeDateLine.csv", prepareWholeDateline, function (data1) {
    datelineArray = data1;
    startDate = datelineArray[0];
    endDate = datelineArray[data1.length - 1];

    sliderDatesScale.domain([startDate, endDate]);

    line = slider.append("line")
        .attr("class", "track")
        .attr("x1", sliderDatesScale.range()[0])
        .attr("x2", sliderDatesScale.range()[1])
        .select(function () {
            return this.parentNode.appendChild(this.cloneNode(true));
        })
        .attr("class", "track-inset")
        .select(function () {
            return this.parentNode.appendChild(this.cloneNode(true));
        })
        .attr("class", "track-overlay")
        .call(d3.drag()
            .on("start.interrupt", function () {
                slider.interrupt();
            })
            .on("start drag", function () {
                currentIndex = getCurrentIndex(d3.event.x);
                update();
            })
        );


    group = slider.insert("g", ".track-overlay")
        .attr("class", "ticks")
        .attr("transform", "translate(0," + 18 + ")");


    handle = slider.insert("circle", ".track-overlay")
        .attr("class", "handle")
        .attr("r", 9)
        .attr("cx", slider_range_start);


    label = slider.append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .text(formatDate(startDate))
        .attr("transform", "translate(" + 0 + "," + (-15) + ")")
        .attr("x", slider_range_start);

    textsSlider = group.selectAll("text")
        .data(sliderDatesScale.ticks(5))
        .enter()
        .append("text")
        .attr("x", sliderDatesScale)
        .attr("y", 10)
        .attr("text-anchor", "middle")
        .text(function (d) {
            return formatDateIntoMonthDayYear(d);
        });


    d3.csv("/static/data/v1/currentDates.csv", prepareDates, function (data2) {
        currentDateHorizonArrays = data2;

        d3.csv("/static/data/v1/futureDates.csv", prepareDates, function (data3) {
            futureDatesArrays = data3;

            d3.csv("/static/data/v1/influenza" + stateString + ".csv", prepareValues, function (data4) {
                currentHorizonInfluenzaArrays = data4;

                d3.csv("/static/data/v1/prediction1" + stateString + ".csv", prepareValues, function (data5) {
                    prediction1InfluArrays = data5;

                    d3.csv("/static/data/v1/prediction2" + stateString + ".csv", prepareValues, function (data6) {
                        prediction2InfluArrays = data6;

                        actualDatesInfluArrays = getZippedDatasets(currentDateHorizonArrays, currentHorizonInfluenzaArrays);
                        pred1DatesInfluArrays = getZippedDatasets(futureDatesArrays, prediction1InfluArrays);
                        pred2DatesInfluArrays = getZippedDatasets(futureDatesArrays, prediction2InfluArrays);


                        actualDatesInfluArray = actualDatesInfluArrays[currentIndex];
                        pred1DatesInfluArray = pred1DatesInfluArrays[currentIndex];
                        pred2DatesInfluArray = pred2DatesInfluArrays[currentIndex];

                        // Setting the max value of the y scale according to the input.
                        var yMaxValue = 7;
                        for (var i = 0; i < currentHorizonInfluenzaArrays.length; i++) {
                            yMaxValue = d3.max([yMaxValue, d3.max(currentHorizonInfluenzaArrays[i])])
                        }
                        yScale.domain([0, yMaxValue]);

                        var domainStartDate = actualDatesInfluArray[0][0];
                        var domainEndDate = actualDatesInfluArray[actualDatesInfluArray.length - 1][0];

                        xScale.domain([domainStartDate, domainEndDate]);


                        //Define X axis
                        xAxis = d3.axisBottom()
                            .scale(xScale)
                            .tickValues(currentDateHorizonArrays[currentIndex])
                            .tickFormat(formatDate);

                        //Define Y axis
                        yAxis = d3.axisLeft()
                            .scale(yScale)
                            .ticks(5);
                        //Create X axis
                        xAxisGroup = svg.append("g")
                            .attr("class", "x_axis")
                            .attr("transform", "translate(0," + topMarginxAxis + ")")
                            .call(xAxis)
                            .selectAll("text")
                            .style("text-anchor", "end")
                            .attr("dx", "-.8em")
                            .attr("dy", ".15em")
                            .attr("transform", "rotate(-40)");

                        //Create Y axis
                        svg.append("g")
                            .attr("class", "y axis")
                            .attr("transform", "translate(" + leftMarginyAxis + ", 0)")
                            .call(yAxis);

                        var currentDate = datelineArray[currentIndex];

                        svg.append("line")
                            .attr("x1", xScale(currentDate))
                            .attr("y1", yScale(0))
                            .attr("x2", xScale(currentDate))
                            .attr("y2", yScale(yMaxValue))
                            .style("stroke-width", 2)
                            .style("stroke", "black")
                            .style("fill", "none")
                            .style("stroke-dasharray", ("3, 3"));

                        lineGraphActual = svg.append("path")
                            .attr("d", lineFunction(actualDatesInfluArray))
                            .attr("stroke", actualValuesColor)
                            .attr("stroke-width", 1)
                            .attr("fill", "none");

                        lineGraphPred1 = svg.append("path")
                            .attr("d", lineFunction(pred1DatesInfluArray))
                            .attr("stroke", pred1Color)
                            .attr("stroke-width", 1)
                            .attr("fill", "none");

                        lineGraphPred2 = svg.append("path")
                            .attr("d", lineFunction(pred2DatesInfluArray))
                            .attr("stroke", pred2Color)
                            .attr("stroke-width", 1)
                            .attr("fill", "none");

                        playButton
                            .on("click", function () {
                                var button = d3.select(this);
                                if (button.text() == "Pause") {
                                    moving = false;
                                    clearInterval(timer);
                                    // timer = 0;
                                    button.text("Play");
                                } else {
                                    moving = true;
                                    timer = setInterval(step, 2000);
                                    button.text("Pause");
                                }
                                console.log("Slider moving: " + moving);
                            });

                    });

                });

            });

        });


    });

});


// This function updates the plot based on the drop down selection.
// The drop down menu offers to choose a state.
function updatePlotBasedOnSelection(stateString) {

    d3.csv("/static/data/v1/wholeDateLine.csv", prepareWholeDateline, function (data1) {

        datelineArray = data1;
        startDate = datelineArray[0];
        endDate = datelineArray[data1.length - 1];

        sliderDatesScale.domain([startDate, endDate]);

        slider
            .attr("x1", sliderDatesScale.range()[0])
            .attr("x2", sliderDatesScale.range()[1])
            .attr("cx", slider_range_start);

        label
            .text(formatDate(startDate))
            .attr("x", slider_range_start);

        textsSlider
            .data(sliderDatesScale.ticks(5))
            .attr("x", sliderDatesScale);


        d3.csv("/static/data/v1/currentDates.csv", prepareDates, function (data2) {
            currentDateHorizonArrays = data2;

            d3.csv("/static/data/v1/futureDates.csv", prepareDates, function (data3) {
                futureDatesArrays = data3;

                d3.csv("/static/data/v1/influenza" + stateString + ".csv", prepareValues, function (data4) {
                    currentHorizonInfluenzaArrays = data4;

                    d3.csv("/static/data/v1/prediction1" + stateString + ".csv", prepareValues, function (data5) {
                        prediction1InfluArrays = data5;

                        d3.csv("/static/data/v1/prediction2" + stateString + ".csv", prepareValues, function (data6) {
                            prediction2InfluArrays = data6;

                            actualDatesInfluArrays = getZippedDatasets(currentDateHorizonArrays, currentHorizonInfluenzaArrays);
                            pred1DatesInfluArrays = getZippedDatasets(futureDatesArrays, prediction1InfluArrays);
                            pred2DatesInfluArrays = getZippedDatasets(futureDatesArrays, prediction2InfluArrays);

                            update();

                        })
                    })
                })
            })
        })
    })
};


function step() {
    currentIndex = currentIndex + 1;
    if (actualDatesInfluArrays.length - 1 < currentIndex ) {
        moving = false;
        currentIndex = 0;
        clearInterval(timer);
        // timer = 0;
        playButton.text("Play");
        console.log("Slider moving: " + moving);
    } else {
        update();
    }
};


// Update the the drag button
function update() {
    // update position and text of label according to slider scale
    handle.attr("cx", sliderDatesScale(datelineArray[currentIndex]));
    label
    .attr("x", sliderDatesScale(datelineArray[currentIndex]))
    .text(formatDate(datelineArray[currentIndex]));

    // Update the plot
    actualDatesInfluArray = actualDatesInfluArrays[currentIndex];
    pred1DatesInfluArray = pred1DatesInfluArrays[currentIndex];
    pred2DatesInfluArray = pred2DatesInfluArrays[currentIndex];

    var domainStartDate = new Date(actualDatesInfluArray[0][0]);
    var domainEndDate = actualDatesInfluArray[actualDatesInfluArray.length-1][0];

    xScale.domain([domainStartDate, domainEndDate ]);

    xAxis.scale(xScale)
        .tickValues(currentDateHorizonArrays[currentIndex]);

    svg.select(".x_axis")
    .call(xAxis)
    .selectAll("text")
        .style("text-anchor", "end")
        .attr("dx", "-.8em")
        .attr("dy", ".15em")
        .attr("transform", "rotate(-40)" );

    var currentDate = datelineArray[currentIndex];

    svg.append("line")
        .attr("x1", xScale(currentDate))
        .attr("x2", xScale(currentDate));

    lineGraphActual
        .transition()
        .duration(0)
        .attr("d", lineFunction(actualDatesInfluArray));

    lineGraphPred1
        .transition()
        .delay(300)
        .duration(800)
        .attr("d", lineFunction(pred1DatesInfluArray));

    lineGraphPred2
        .transition()
        .delay(300)
        .duration(1000)
        .attr("d", lineFunction(pred2DatesInfluArray));

};
