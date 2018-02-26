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
var pred1Color = "#036564";  // "#ff661a";
var pred2Color = "#550b1d";  // "#ff3333";


var playButton = d3.select("#play-button");


// Drop Down Selection Menus:
var options = ['Baden-Wuerttemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg', 'Hessen',
                   'Mecklenburg-Vorpommern', 'Niedersachsen', 'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland',
                   'Sachsen', 'Sachsen-Anhalt', 'Schleswig-Holstein', 'Thueringen'];
var stateString = options[0];

var optionsYear = ['2014', '2013', '2012', '2011', '2010', '2008', '2007', '2006', '2005']
var year ='2014';

var select = d3.select("#fpsDD")
    .on('change',onchange);

var selectYear = d3.select("#fpsDDYear")
    .on('change',onchange);

select
  .selectAll('option')
	.data(options).enter()
	.append('option')
		.text(function (d) { return d; });

selectYear
  .selectAll('option')
	.data(optionsYear).enter()
	.append('option')
		.text(function (d) { return d; });

function onchange() {
	selectValue = d3.select('select').property('value');
	selectValueYear = d3.select('#fpsDDYear').property('value');
	updatePlotBasedOnSelection(selectValue, selectValueYear);
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
   .text("# Reported Influenza Infections per 100 000 Inhabitants")
   .attr("text-anchor", "middle")
   .attr("font-family", "sans-serif")
   .attr("font-size", axisLabelTextSize)
   .attr("fill", "black")
    .attr("transform", "translate(" + (leftMarginyAxis - 30) + "," + Math.round((yScaleRangeEnd - yScaleRangeStart)/2 + yScaleRangeStart) + ")rotate(-90)");

// Legend
var xlegendLineStart = xScaleRangeStart + (xScaleRangeEnd - xScaleRangeStart)/40 //svgWidth * 7/9 - 25;
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

var areaGraphActual;
var areaGraphPred1;
var areaGraphPred2;

var moving = false;
var currentIndex = 15;


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

// Area function for the area under the step plot
var area = d3.area()
    .x(function(d) { return xScale(d[0]); })
    .y0(yScaleRangeStart)
    .y1(function(d) { return yScale(d[1]); })
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

d3.csv("/static/data/fpslide/wholeDateLine" + year + ".csv", prepareWholeDateline, function (data1) {
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
        .attr("cx", sliderDatesScale(datelineArray[currentIndex])); // slider_range_start is default


    label = slider.append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .text(formatDate(datelineArray[currentIndex])) // startDate is default
        .attr("transform", "translate(" + 0 + "," + (-15) + ")")
        .attr("x", sliderDatesScale(datelineArray[currentIndex])); // slider_range_start is default

    // Array length to calculate the ticks
    var array_length = datelineArray.length;

    textsSlider = group.selectAll("text")
        .data([datelineArray[0], datelineArray[Math.floor((array_length-1)/3)], datelineArray[Math.floor((array_length-1)*2/3)], datelineArray[array_length-1]])
        .enter()
        .append("text")
        .attr("x", sliderDatesScale)
        .attr("y", 10)
        .attr("text-anchor", "middle")
        .text(function (d) {
            return formatDateIntoMonthDayYear(d);
        });


    d3.csv("/static/data/fpslide/currentDates" + year + ".csv", prepareDates, function (data2) {
        currentDateHorizonArrays = data2;

        d3.csv("/static/data/fpslide/futureDates" + year + ".csv", prepareDates, function (data3) {
            futureDatesArrays = data3;

            d3.csv("/static/data/fpslide/influenza" + stateString + year + ".csv", prepareValues, function (data4) {
                currentHorizonInfluenzaArrays = data4;

                d3.csv("/static/data/fpslide/prediction1" + stateString + year + ".csv", prepareValues, function (data5) {
                    prediction1InfluArrays = data5;

                    d3.csv("/static/data/fpslide/prediction2" + stateString + year + ".csv", prepareValues, function (data6) {
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

                        var dotted_line_text_color = "black";

                         svg.append("text")
                           .text("Past")
                           .attr("text-anchor", "left")
                           .attr("x", xScale(currentDate) - (xScaleRangeEnd - xScaleRangeStart)/10)
                           .attr("y", yScale(yMaxValue))
                           .attr("font-family", "sans-serif")
                           .attr("font-size", textSize)
                           .attr("fill", dotted_line_text_color);

                        svg.append("line")
                            .attr("x1", xScale(currentDate))
                            .attr("y1", yScale(0))
                            .attr("x2", xScale(currentDate))
                            .attr("y2", yScale(yMaxValue))
                            .style("stroke-width", 2)
                            .style("stroke", "black")
                            .style("fill", "none")
                            .style("stroke-dasharray", ("3, 3"));

                        svg.append("text")
                           .text(" Future")
                           .attr("text-anchor", "left")
                           .attr("x", xScale(currentDate) + (xScaleRangeEnd - xScaleRangeStart)/20)
                           .attr("y", yScale(yMaxValue))
                           .attr("font-family", "sans-serif")
                           .attr("font-size", textSize)
                           .attr("fill", dotted_line_text_color);

                        svg.append("line")
                            .attr("x1", xScale(datelineArray[currentIndex+4]))
                            .attr("y1", yScale(0))
                            .attr("x2", xScale(datelineArray[currentIndex+4]))
                            .attr("y2", yScale(yMaxValue))
                            .style("stroke-width", 2)
                            .style("stroke", "black")
                            .style("fill", "none")
                            .style("stroke-dasharray", ("3, 7"));

                        svg.append("text")
                           .text(" More than 4 Weeks in Advance")
                           .attr("text-anchor", "left")
                           .attr("x", xScale(datelineArray[currentIndex+4]) + (xScaleRangeEnd - xScaleRangeStart)/40)
                           .attr("y", yScale(yMaxValue))
                           .attr("font-family", "sans-serif")
                           .attr("font-size", textSize)
                           .attr("fill", dotted_line_text_color);

                        svg.append("line")
                            .attr("x1", xScale(datelineArray[currentIndex+10]))
                            .attr("y1", yScale(0))
                            .attr("x2", xScale(datelineArray[currentIndex+10]))
                            .attr("y2", yScale(yMaxValue))
                            .style("stroke-width", 2)
                            .style("stroke", "black")
                            .style("fill", "none")
                            .style("stroke-dasharray", ("3, 11"));

                        svg.append("text")
                           .text(" More than 10 Weeks in Advance")
                           .attr("text-anchor", "left")
                           .attr("x", xScale(datelineArray[currentIndex+10]) + (xScaleRangeEnd - xScaleRangeStart)/40)
                           .attr("y", yScale(yMaxValue))
                           .attr("font-family", "sans-serif")
                           .attr("font-size", textSize)
                           .attr("fill", dotted_line_text_color);

                        lineGraphActual = svg.append("path")
                            .attr("d", lineFunction(actualDatesInfluArray))
                            .attr("stroke", actualValuesColor)
                            .attr("stroke-width", 1)
                            .attr("fill", "none");

                        areaGraphActual = svg.append("path")
                           .attr("d", area(actualDatesInfluArray))
                           .attr("class", "areaActual");

                        lineGraphPred1 = svg.append("path")
                            .attr("d", lineFunction(pred1DatesInfluArray))
                            .attr("stroke", pred1Color)
                            .attr("stroke-width", 1)
                            .attr("fill", "none");

                        areaGraphPred1 = svg.append("path")
                           .attr("d", area(pred1DatesInfluArray))
                           .attr("class", "areaPred1");

                        lineGraphPred2 = svg.append("path")
                            .attr("d", lineFunction(pred2DatesInfluArray))
                            .attr("stroke", pred2Color)
                            .attr("stroke-width", 1)
                            .attr("fill", "none");

                        areaGraphPred2 = svg.append("path")
                           .attr("d", area(pred2DatesInfluArray))
                           .attr("class", "areaPred2");

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
function updatePlotBasedOnSelection(stateString, yearString) {

    d3.csv("/static/data/fpslide/wholeDateLine" + yearString + ".csv", prepareWholeDateline, function (data1) {

        datelineArray = data1;
        startDate = datelineArray[0];
        endDate = datelineArray[data1.length - 1];

        sliderDatesScale.domain([startDate, endDate])
                        .range([slider_range_start, slider_range_end]);

        slider
            .attr("x1", sliderDatesScale.range()[0])
            .attr("x2", sliderDatesScale.range()[1])
            .attr("cx", slider_range_start);

        // Array length to calculate the ticks
        var array_length = datelineArray.length

        textsSlider
            .data([datelineArray[0], datelineArray[Math.floor((array_length-1)/3)], datelineArray[Math.floor((array_length-1)*2/3)], datelineArray[array_length-1]])
            .attr("x", sliderDatesScale)
            .attr("y", 10)
            .attr("text-anchor", "middle")
            .text(function (d) {
                return formatDateIntoMonthDayYear(d);
            });


        d3.csv("/static/data/fpslide/currentDates" + yearString + ".csv", prepareDates, function (data2) {
            currentDateHorizonArrays = data2;

            d3.csv("/static/data/fpslide/futureDates" + yearString + ".csv", prepareDates, function (data3) {
                futureDatesArrays = data3;

                d3.csv("/static/data/fpslide/influenza" + stateString + yearString + ".csv", prepareValues, function (data4) {
                    currentHorizonInfluenzaArrays = data4;

                    d3.csv("/static/data/fpslide/prediction1" + stateString + yearString + ".csv", prepareValues, function (data5) {
                        prediction1InfluArrays = data5;

                        d3.csv("/static/data/fpslide/prediction2" + stateString + yearString + ".csv", prepareValues, function (data6) {
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

    // In case the svg is updated and the current happens to be out of bound with respect
    // new array.
    if (actualDatesInfluArrays.length - 1 < currentIndex) {
        currentIndex = actualDatesInfluArrays.length - 1
    }

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

    areaGraphActual
        .transition()
        .duration(0)
        .attr("d", area(actualDatesInfluArray));

    lineGraphPred1
        .transition()
        .delay(300)
        .duration(800)
        .attr("d", lineFunction(pred1DatesInfluArray));

    areaGraphPred1
        .transition()
        .delay(300)
        .duration(800)
        .attr("d", area(pred1DatesInfluArray));

    lineGraphPred2
        .transition()
        .delay(300)
        .duration(1000)
        .attr("d", lineFunction(pred2DatesInfluArray));

    areaGraphPred2
        .transition()
        .delay(300)
        .duration(1000)
        .attr("d", area(pred2DatesInfluArray));

};

