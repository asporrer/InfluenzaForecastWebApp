import os
from collections import OrderedDict, defaultdict
from isoweek import Week

import pandas as pd
import numpy as np

import dill as pickle

from sklearn.neighbors import KernelDensity

# Plotting
from bokeh.plotting import figure
from bokeh.models import Range1d, LinearAxis, DatetimeTickFormatter, ColumnDataSource, HoverTool
from bokeh.palettes import Category20b
from bokeh.transform import linear_cmap
from bokeh.layouts import column, row
from bokeh.embed import components

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from flask_moment import Moment

# Paths
currDir = os.path.dirname(__file__)
data_dir = os.path.join(currDir, "static")
data_dir = os.path.join(data_dir, "data")
data_location = os.path.join(data_dir, 'datadf.pkl')
data_for_metric_location = os.path.join(data_dir, 'influenzaDataForMetric.pkl')
data_plots_location = os.path.join(data_dir, "plots")

# Page 2 plots
vanilla_script_div_path = os.path.join(data_plots_location, "vanilla_script_div.pkl")
overall_cases_script_div_path = os.path.join(data_plots_location, "overall_cases_script_div.pkl")
wave_stats_script_div_path = os.path.join(data_plots_location, "wave_stats_script_div.pkl")
wave_start_vs_intensity_script_div_path = os.path.join(data_plots_location, "wave_start_vs_intensity_script_div.pkl")

# Page 3 plots
features_script_div_path = os.path.join(data_plots_location, "features_script_div.pkl")
data_plots_metric_location = os.path.join(data_plots_location, "metrics")
metrics_auc_script_div_path = os.path.join(data_plots_metric_location, "metric_AUC_script_div.pkl")
metrics_accuracy_script_div_path = os.path.join(data_plots_metric_location, "metric_Accuracy_script_div.pkl")
metrics_f2score_script_div_path = os.path.join(data_plots_metric_location, "metric_F2Score_script_div.pkl")
metrics_logloss_script_div_path = os.path.join(data_plots_metric_location, "metric_LogLoss_script_div.pkl")
metrics_precision_script_div_path = os.path.join(data_plots_metric_location, "metric_Precision_script_div.pkl")
metrics_recall_script_div_path = os.path.join(data_plots_metric_location, "metric_Recall_script_div.pkl")
conf_mat_thr1_script_div_path = os.path.join(data_plots_metric_location, "confMatPlotTh1.pkl")
conf_mat_thr2_script_div_path = os.path.join(data_plots_metric_location, "confMatPlotTh2.pkl")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'd90Fj238A679bn940sn4Ghrq9b08a962Nvfm2390'

bootstrap = Bootstrap(app)
moment = Moment(app)

# Loading the data frame containing the features and target variable for all sixteen states from 2005 until 2015.
with open(data_location, 'rb') as file:
    data_df = pickle.load(file)

# Used for selecting and plotting metrics figures. The order of the lists matters.
metric_name_list = ['Accuracy', 'Precision', 'Recall', 'F2_Score', 'ROC_AUC', 'Log_Loss']
metric_path_list = [metrics_accuracy_script_div_path, metrics_precision_script_div_path, metrics_recall_script_div_path,
                    metrics_f2score_script_div_path, metrics_auc_script_div_path, metrics_logloss_script_div_path]


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/', methods=['GET'])
def index():
    """
    The start page of the influenza project is rendered.
    More information about the d3 visualization can be found in the fpsslide.css file.

    :return: A str, containing the respective html and javascript code.
    """
    return render_template('index.html')


@app.route('/InfluenzaProject1', methods=['GET'])
def render_influenza_project1():
    return render_template('index.html')


@app.route('/InfluenzaProject2', methods=['GET'])
def render_influenza_project2():
    """
    This function renders the second page of the influenza project.
    To improve loading time pre-generated figures are loaded.

    :return: A str, containing the respective html and javascript code.
    """

    state_list = data_df['state'].unique().tolist()

    # Loading pregenerated plots to speed up loading of the page.
    with open(vanilla_script_div_path, 'rb') as file:
        script_vanilla, div_vanilla = pickle.load(file)

    with open(overall_cases_script_div_path, 'rb') as file:
        script_overall_cases, div_overall_cases = pickle.load(file)

    with open(wave_stats_script_div_path, 'rb') as file:
        script_wave_stats, div_wave_stats = pickle.load(file)

    with open(wave_start_vs_intensity_script_div_path, 'rb') as file:
        script_wave_start_vs_intensity, div_wave_start_vs_intensity = pickle.load(file)

    return render_template('InfluenzaProject2.html', script_vanilla=script_vanilla, div_vanilla=div_vanilla,
                           script_overall_cases=script_overall_cases, div_overall_cases=div_overall_cases,
                           script_wave_stats=script_wave_stats, div_wave_stats=div_wave_stats,
                           script_wave_start_vs_intensity=script_wave_start_vs_intensity,
                           div_wave_start_vs_intensity=div_wave_start_vs_intensity,
                           stateSequence=state_list)


@app.route('/InfluenzaProject3', methods=['GET'])
def render_influenza_project3():
    """
    This function renders the third page of the influenza project.
    To improve loading time pre-generated figures are loaded.

    :return: A str, containing the respective html and javascript code.
    """

    # Loading pregenerated plots to speed up loading of the page.
    with open(features_script_div_path, 'rb') as file:
        script_features, div_features = pickle.load(file)

    with open(metrics_auc_script_div_path, 'rb') as file:
        script_metric, div_metric = pickle.load(file)

    with open(conf_mat_thr1_script_div_path, 'rb') as file:
        script_conf_mat_thr1, div_conf_mat_thr1 = pickle.load(file)

    with open(conf_mat_thr2_script_div_path, 'rb') as file:
        script_conf_mat_thr2, div_conf_mat_thr2 = pickle.load(file)

    state_list = data_df['state'].unique().tolist()

    return render_template('InfluenzaProject3.html',
                           script_features=script_features,
                           div_features=div_features,
                           script_metric=script_metric,
                           div_metric=div_metric,
                           script_conf_mat_thr1=script_conf_mat_thr1,
                           div_conf_mat_thr1=div_conf_mat_thr1,
                           script_conf_mat_thr2=script_conf_mat_thr2,
                           div_conf_mat_thr2=div_conf_mat_thr2,
                           stateSequence=state_list,
                           metricSequence=metric_name_list)


@app.route('/Contact', methods=['GET'])
def render_contact():
    """
    The contact page of the influenza project is rendered.

    :return: A str, containing the respective html and javascript code.
    """

    return render_template('contact.html')


@app.route('/waveStatisticsFigure')
def wave_statistics_figure():
    """
    This function is called via an ajax. The code is located at ajaxscripts.js.

    :return: A str, containing the respective html and javascript code.
    """

    state_list = request.args.getlist('jsdata[]')

    p_wave_stats = visualize_wave_stats_distributions(data_df, states=state_list)

    script, div = components(p_wave_stats)

    return render_template('multiselectUpdatedFigure.html', script=script, div=div)


@app.route('/waveStartVsIntensityFigure')
def wave_start_vs_intensity_figure():
    """
    This function is called via an ajax. The code is located in ajaxscripts.js.

    :return: A str, containing the respective html and javascript code.
    """

    state_list = request.args.getlist('jsdata[]')
    p_start_vs_severity = visualize_wave_start_vs_severity_via_box(data_df, states=state_list)
    p_start_vs_length = visualize_wave_start_vs_length_via_box(data_df, states=state_list)

    script, div = components(row(p_start_vs_severity, p_start_vs_length))

    return render_template('multiselectUpdatedFigure.html', script=script, div=div)


@app.route('/featuresFigure')
def features_figure():
    """
    This function is called via an ajax. The code is located in ajaxscripts.js.

    :return: A str, containing the respective html and javascript code.
    """

    state_str = request.args['jsdata']

    p_wave_features = visualize_data_per_state(data_df, state_str=state_str)

    script, div = components(p_wave_features)

    return render_template('multiselectUpdatedFigure.html', script=script, div=div)


@app.route('/metricFigure')
def metrics_figure():
    """
    This function is called via an ajax. The code is located in ajaxscripts.js.

    :return: A str, containing the respective html and javascript code.
    """

    state_str = request.args['jsdata']

    for index_metric, name_metric in enumerate(metric_name_list):
        if state_str == name_metric:
            # Loading pre-generated plots to speed up loading of the page.
            with open(metric_path_list[index_metric], 'rb') as file:
                script, div = pickle.load(file)

    return render_template('multiselectUpdatedFigure.html', script=script, div=div)


#######################
# Visualization methods
#######################

def generate_new_plots():
    """
    This function generates new versions of the bokeh plots
    transforms them into an embeddable format and than stores them
    to data/plots. Reloading the stored plots improves the loading
    time as compared to generating the plots from scratch.

    :return: None
    """

    # Generate figures
    p_vanilla_influenza = visualize_state_commonalities(data_df)
    p_overall_reported_cases = visualize_overall_reported_cases(data_df)
    p_wave_stats = visualize_wave_stats_distributions(data_df)
    p_start_vs_severity = visualize_wave_start_vs_severity_via_box(data_df)
    p_start_vs_length = visualize_wave_start_vs_length_via_box(data_df)
    p_wave_features = visualize_data_per_state(data_df)

    # Embeddable version of figures for flask
    script_vanilla, div_vanilla = components(p_vanilla_influenza)
    script_overall_cases, div_overall_cases = components(p_overall_reported_cases)
    script_wave_stats, div_wave_stats = components(p_wave_stats)
    script_wave_start_vs_intensity, div_wave_start_vs_intensity = components(row(p_start_vs_severity, p_start_vs_length))
    script_features, div_features = components(p_wave_features)

    # Storing new plots
    with open(vanilla_script_div_path, 'wb') as file:
        pickle.dump((script_vanilla, div_vanilla), file)

    with open(overall_cases_script_div_path, 'wb') as file:
        pickle.dump((script_overall_cases, div_overall_cases), file)

    with open(wave_stats_script_div_path, 'wb') as file:
        pickle.dump((script_wave_stats, div_wave_stats), file)

    with open(wave_start_vs_intensity_script_div_path, 'wb') as file:
        pickle.dump((script_wave_start_vs_intensity, div_wave_start_vs_intensity), file)

    with open(features_script_div_path, 'wb') as file:
        pickle.dump((script_features, div_features), file)


def visualize_state_commonalities(data_df):
    """
    This function returns a bokeh visualization of the influenza waves for the states of Germany from 2005 till 2015.

    :param data_df: A pandas.DataFrame, containing the influenza progression of the sixteen states of Germany.
    :return: A bokeh figure, visualizing the influenza progressions of the sixteen states of Germany.
    """

    kwargs_state_influenza_dict = OrderedDict()

    for current_state in data_df['state'].unique():

            # State information
            state_indices = data_df['state'] == current_state
            state_df = data_df[state_indices]

            kwargs_state_influenza_dict[current_state] = state_df['influenza_week-1'].tolist()[:-1]

    return plot_results('States of Germany', 'Date', '# Influenza Infections per 100 000 Inhabitants',
                        data_df['year_week'].unique().tolist()[1:], **kwargs_state_influenza_dict)


def plot_results(title_param, x_axis_title_param, y_axis_title_param, dates_list=None, **kwargs):
    """
    A generic function for plotting step functions.

    :param title_param: A str, the title of the plot.
    :param x_axis_title_param:  A str, the x axis title.
    :param y_axis_title_param: A str, the y axis title.
    :param dates_list: A list of dates, used to scale and format the x axis accordingly.
    :param kwargs: A list of float or int, representing the y coordinates.
    :return: A bokeh figure, of a step function plot.
    """

    if kwargs is None:
        raise Exception('kwargs is not allowed to be None.')

    p = figure(plot_width=800, plot_height=500, title=title_param, x_axis_label=x_axis_title_param,
               y_axis_label=y_axis_title_param, toolbar_location="right")


    # In case a dates list is provided as parameter:
    # This list is formatted and then used in the plot.
    if dates_list:
        plot_x = [Week(year_week[0], year_week[1]).monday() for year_week in dates_list]
        p.xaxis.formatter = DatetimeTickFormatter(years=["%D %B %Y"])
        p.xaxis.major_label_orientation = 3.0 / 4
        p.xaxis[0].ticker.desired_num_ticks = 30

    # Coloring different graphs
    color_list = Category20b[16]
    number_of_colors = len(color_list)
    color_index = 0

    for key, argument in kwargs.items():
        if not dates_list:
            plot_x = range(len(argument))

        color_index += 1
        p.step(plot_x, argument, legend=key, color=color_list[color_index % number_of_colors], alpha=1.0,
               muted_color=color_list[color_index % number_of_colors], muted_alpha=0.0)

    p.legend.location = "top_left"
    p.legend.click_policy="mute"
    p.legend.background_fill_alpha = 0.0

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    return p


def visualize_overall_reported_cases(input_df):
    """
    This function visualizes the cumulative number of reported influenza infections for each of the sixteen
    states of Germany from 2005 till 2015. The period from the 25th week of 2009 till the 24th week of 2010 is excluded.
    In this period the "outlier wave" (swine flu) occurred.

    :param input_df: A pandas.DataFrame, containing a rows with names 'state', 'influenza_week-1'.
    :return: A bokeh figure, visualizing the sum of the overall reported influenza infections for each of the sixteen
    states of Germany.
    """
    input_list = get_number_of_reported_infected_per_state(input_df)

    states = [state_sum[0] for state_sum in input_list]
    sum_of_rep_cases = [state_sum[1] for state_sum in input_list]

    source = ColumnDataSource(data=dict(states=states, sum_of_rep_cases=sum_of_rep_cases))

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    p = figure(plot_height=500, plot_width=500, x_range=states, y_range=(0, 900), tools=TOOLS,
               title="Total Number of Reported Influenza Infections from 2005-2015",
               y_axis_label='# Reported Influenza Infections per 100 000 inhabitants')

    p.vbar(x='states', top='sum_of_rep_cases', width=0.9, source=source, legend=False,
           line_color='white', fill_color=linear_cmap(field_name='sum_of_rep_cases', palette=["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"], low=0,
                                                      high=900))
    p.line(x=[0, 16], y=[500, 500], color='black')
    p.line(x=[0, 16], y=[300, 300], color='black')

    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 3.0 / 4
    p.ygrid.grid_line_color = None

    return p


def get_number_of_reported_infected_per_state(input_df):
    """
    This function extracts the overall number of reported influenza infections for each state in Germany and returns
    the respective dictionary. The period from the 25th week of 2009 till the 24th week of 2010 is excluded. In this
    period the "outlier wave" (swine flu) occurred.

    :param input_df: A pandas.DataFrame, containing a rows with names 'state', 'influenza_week-1'.
    :return: A dict, holding the sixteen states of Germany as key. The values are the overall sum of reported influenza
    cases normalized by 100 000 inhabitants.
    """

    start_year_excluded = 2009
    start_week_excluded = 25
    end_year_excluded = 2010
    end_week_excluded = 24

    interval_bool_series = input_df['year_week'].apply(
        lambda x: not((start_year_excluded <= x[0] and (start_week_excluded <= x[1] or start_year_excluded < x[0])) and (
                x[0] <= end_year_excluded and (x[1] <= end_week_excluded or x[0] < end_year_excluded))))

    return sorted(list(input_df[interval_bool_series].groupby('state')['influenza_week-1'].sum().to_dict().items()),
                  key=lambda x: x[1])


def visualize_wave_stats_distributions(input_df, states=['all']):
    """
    This function provides statistics about the influenza wave on a state level in the form of a figure.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', 'influenza_week-1'.
    :param states: A list, containing the names of the states relevant for the statistic.
    :return: A Bokeh Column, containing the four figures visualizing wave statistics.
    """

    # Getting the wave statistics.
    first_last_week_max_length_of_wave_dict = get_first_last_week_max_length_of_wave(input_df, current_states=states)

    first_list = [year_week[1] for year_week in first_last_week_max_length_of_wave_dict['first']]
    last_list = [year_week[1] for year_week in first_last_week_max_length_of_wave_dict['last']]
    max_list = first_last_week_max_length_of_wave_dict['max']
    length_list = first_last_week_max_length_of_wave_dict['length']

    value_list_list = [first_list, last_list, max_list, length_list]
    title_list = ['Week of Year the Wave Started', 'Week of Year the Wave Ended',
                  'Severity of the Wave in Infected per 100 000 Inhabitants', 'Duration of the Wave in Weeks']
    x_labels_list = ['Week of the Year', 'Week of the Year', 'Infected per 100 000 Inhabitants', 'Duration in Weeks']
    p_list = []

    # Generating the histogram and estimated density.
    for index in range(4):
        min_value = min(value_list_list[index])
        max_value = max(value_list_list[index])
        num_of_bins = int(max_value-min_value)

        hist, edges = np.histogram(value_list_list[index], density=True,
                                   bins=num_of_bins)

        p = figure(title=title_list[index], x_axis_label=x_labels_list[index], y_axis_label='Probabilities',
                   plot_width=400, plot_height=350,)

        p.xaxis[0].ticker.desired_num_ticks = 20

        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="#036564", line_color="#033649")

        x_gird = np.linspace(start=min_value-1, stop=max_value+1, num=2*num_of_bins)
        p.line(np.array(x_gird), kde_sklearn(np.array(value_list_list[index]), np.array(x_gird), bandwidth=1.5), color='black')
        p_list.append(p)

    row1 = row(p_list[0], p_list[1])
    row2 = row(p_list[2], p_list[3])

    return column(row1, row2)


def get_first_last_week_max_length_of_wave(input_df, current_states=['all']):
    """
    This function returns the wave start, wave end, wave length and height of the states specified by the above state
    parameter. A wave start is defined as the first week of a cold weather season in which the threshold of 2.0 infected
    per 100 000 inhabitants is crossed. The outlier year 2009 (swine flu) is excluded.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', 'influenza_week-1'.
    :param current_states: A list, containing the names of the states relevant for the statistic.
    :return: A dict, containing the first, last week of a wave, the length and the height of a wave in the form of
    four numpy.ndarrays. First, last week and length, height are associated via the index of their numpy.ndarray.
    That is to say same index means that the value belongs to the same wave.
    """

    first_list = []
    last_list = []
    max_list = []

    # Extracting the wave start, end and height from the data frame.
    for state in input_df['state'].unique().tolist():
        if state in current_states or current_states[0] == 'all':
            helper_first_list, helper_last_list, helper_max_list = get_first_last_max_per_state(input_df, state)
            first_list.extend(helper_first_list)
            last_list.extend(helper_last_list)
            max_list.extend(helper_max_list)

    # Calculating the wave lengths.
    length_list = []
    for index in range(len(first_list)):
        if last_list[index][1] < first_list[index][1]:
            length_list.append(last_list[index][1] + 53 - first_list[index][1])
        else:
            length_list.append(last_list[index][1] - first_list[index][1] + 1)

    # Creating boolean index implementing the specified selection.
    no_2009_and_not_waveless_indices = []
    for index in range(len(first_list)):
        current_year_week = first_list[index]
        if (current_year_week[0] != 2009 or current_year_week[1] < 25) and 0 < current_year_week[1]:
            no_2009_and_not_waveless_indices.append(True)
        else:
            no_2009_and_not_waveless_indices.append(False)

    return {'first': list(np.array(first_list)[no_2009_and_not_waveless_indices]),
            'last': list(np.array(last_list)[no_2009_and_not_waveless_indices]),
            'max': list(np.array(max_list)[no_2009_and_not_waveless_indices]),
            'length': list(np.array(length_list)[no_2009_and_not_waveless_indices])}


def get_first_last_max_per_state(input_df, state_str, start_year=2005, end_year=2015, start_week=25, end_week=24,
                                 target_column_name='influenza_week-1', threshold=2.0):
    """
    This function returns the wave start, wave end, wave max of the states specified by the above state
    parameter. The underlying quantity does not have to be the number of influenza cases on a state level and it is
    specified by target column parameter. In the usual case of referring to influenza infections a wave start is defined
    as the first week of a cold weather season in which the threshold of infected per 100 000 inhabitants is crossed.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', target_column_name.
    :param state_str: A str, the name of the state of interest.
    :param start_year: An int, the start year of the interval in which the start, end and max of the waves are
    calculated.
    :param end_year: An int, the end year of the interval in which the start, end and max of the waves are
    calculated.
    :param start_week: An int, the start week of the interval in which the start, end and max of the waves are
    calculated.
    :param end_week: An int, the end week of the interval in which the start, end and max of the waves are
    calculated.
    :param target_column_name: The target column specifying the quantity of interest.
    :param threshold: A float, the threshold specifying the start and the end of a wave.
    :return: A 3-tuple, containing the first, last week of a wave and the max of a wave in the form of three lists.
    First, last week and max are associated via the list index. That is to say same index means that the value belongs
    to the same wave.
    """

    if not ('year_week' in list(input_df.columns) and 'state' in list(input_df.columns)
            and target_column_name in list(input_df.columns)):
        raise ValueError('The input data frame has to have the column names "year_week" and ' + target_column_name
                         + '.')

    first_list = []
    last_list = []
    max_list = []

    state_df = input_df[input_df['state'] == state_str]
    for year in range(start_year, end_year):
        dummy_df, current_year_df = get_wave_complement_interval_split(state_df, start_year=year, start_week=start_week,
                                                                       end_year=year+1, end_week=end_week)

        max_list.append(current_year_df[target_column_name].max())

        influenza_list = current_year_df[target_column_name].tolist()
        first_index = get_first_or_last_greater_than(influenza_list, threshold)
        last_index = get_first_or_last_greater_than(influenza_list, threshold, first=False)

        if first_index is not None:
            helper_first = current_year_df['year_week'].iloc[first_index]
            helper_last = current_year_df['year_week'].iloc[last_index]

            first_list.append((helper_first[0], helper_first[1] - 1))
            last_list.append((helper_last[0], helper_last[1] - 1))
        else:
            first_list.append((year, -1))
            last_list.append((year, -1))

    return first_list, last_list, max_list


def get_wave_complement_interval_split(input_df, start_year, start_week, end_year, end_week):
    """
    This functions splits the input_df into two data frames according to the interval and
    its complement specified by the parameters. (This function can be used
    to perform a train - test split for instance.)

    :param input_df: A pandas.DataFrame, containing a row with the name 'year_week'
    :param start_year: An int, specifying the year the interval starts.
    :param start_week: An int, specifying the week the interval starts.
    :param end_year: An int, specifying the year the interval ends.
    :param end_week: An int, specifying the week the interval ends.
    :return: A 2-tuple, each entry of type pandas.DataFrame. The second data frame contains the rows of the input data
    frame associated with the time interval specified by the start year and week and end year and week. The first data
    frame contains the rows associated with the complement of the above time interval.
    """
    interval_bool_series = input_df['year_week'].apply(
        lambda x: (start_year <= x[0] and (start_week <= x[1] or start_year < x[0])) and (
                x[0] <= end_year and (x[1] <= end_week or x[0] < end_year)))

    interval_complement_bool_series = interval_bool_series.apply(
        lambda x: not x)

    interval_df = input_df[interval_bool_series]
    interval_complement_df = input_df[interval_complement_bool_series]

    return interval_complement_df, interval_df


def get_first_or_last_greater_than(input_list, threshold_float=2.0, first=True):
    """
    This function takes an list as input and returns the first (or last if first == False) index for which the
    associated list element is greater than the threshold.

    :param input_list: A list, of floats.
    :param threshold_float: A float, the threshold.
    :param first: A bool, specifying whether the index of the first or last element which crosses the threshold is
    returned.
    :return: An int or None, the index of the first (or last in case first == False) list element which crosses the
    threshold. If all elements are smaller or equal to the threshold None is returned.
    """
    my_range = range(len(input_list))

    if not first:
        my_range = reversed(my_range)

    for index in my_range:
        if threshold_float < input_list[index]:
            return index

    return None


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """
    This function returns the estimated density associated to the samples in x. Kernel density estimation is used to
    estimate the density.

    :param x: A numpy.ndarray, containing the samples.
    :param x_grid: A numpy.ndarray, containing the x coordinates at which the estimated density is evaluated.
    :param bandwidth: A float, the bandwidth for the density estimation.
    :param kwargs:
    :return: A numpy.ndarray, the values of the estimated density when evaluated at the grid points.
    """
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)

    kde_skl.fit(x[:, np.newaxis])

    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])

    return np.exp(log_pdf)


def visualize_wave_start_vs_severity_via_box(input_df, states=['all']):
    """
    The relation between the wave start and the severity of the wave is visualized via a box plot.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', 'influenza_week-1'.
    :param states: A list, containing the relevant state names.
    :return: A bokeh figure, visualizing the relation between wave start week and severity via a box plot.
    """

    first_last_max_length_lists = list(get_first_last_week_max_length_of_wave(input_df, current_states=states).items())
    first_list = first_last_max_length_lists[0][1]

    week_list = ["Week " + str(year_week[1]) for year_week in first_list]
    max_value_of_wave_list = first_last_max_length_lists[2][1]

    week_categories_unique_list = sorted(list(set(week_list)))
    week_categories_display_order_list = sorted(week_categories_unique_list, key=lambda x: int(x[-2:].strip()))

    return box_plot(week_list, max_value_of_wave_list, week_categories_display_order_list,
                    title="Wave Start vs Wave Severity", x_axis_label="Calender Week",
                    y_axis_label='Number of Infected per 100 000 Inhabitants')


def visualize_wave_start_vs_length_via_box(input_df, states=['all']):
    """
    The relation between the wave start and the length of the wave is visualized via a box plot.

    :param input_df: A pandas.DataFrame, containing rows with names 'year_week', 'state', 'influenza_week-1'.
    :param states: A list, containing the relevant state names.
    :return: A bokeh figure, visualizing the relation between wave start week and length via a box plot.
    """

    first_last_max_length_lists = list(get_first_last_week_max_length_of_wave(input_df, current_states=states).items())

    start_length_count_df = get_first_length_count_df(first_last_max_length_lists[0][1],
                                                      first_last_max_length_lists[3][1])

    x_count_list = ["Week " + str(week) for week in start_length_count_df['first_week'].tolist()]
    y_count_list = start_length_count_df['wave_length'].tolist()
    count_for_size_list = start_length_count_df['count'].tolist()

    first_week_list = ["Week " + str(year_week[1]) for year_week in first_last_max_length_lists[0][1]]
    wave_length_list = first_last_max_length_lists[3][1]

    week_categories_unique_list = sorted(list(set(first_week_list)))
    week_categories_display_order_list = sorted(week_categories_unique_list, key=lambda x: int(x[-2:].strip()))

    box_fig = box_plot(first_week_list, wave_length_list, week_categories_display_order_list,
                       title="Wave Start vs Wave Length in Weeks", x_axis_label="Calender Week",
                       y_axis_label="Calender Week")
    # Encoding the count of the start week, wave length pair in the x size.
    box_fig.x(x_count_list, y_count_list, color='black', size=np.array(count_for_size_list)*3)

    return box_fig


def box_plot(x_list, y_list, x_cat_unique_display_order_list, title="", x_axis_label="", y_axis_label=""):
    """
    This function returns a box plot. This is a modified version of
    https://bokeh.pydata.org/en/latest/docs/gallery/boxplot.html.

    :param x_list: A list, containing the categories.
    :param y_list: A list, containing the values associated to the categories.
    :param x_cat_unique_display_order_list: A list, containing the categories in a specific order. The categories will
    be displayed on the x-axis in this order.
    :param title: A str, the title of the figure.
    :param x_axis_label: A str, the x axis label.
    :param y_axis_label: A str, the y axis label.
    :return: A bokeh figure, a box plot.
    """

    x_categories_unique_sorted_list = sorted(list(set(x_list)))
    first_max_df = pd.DataFrame(columns=['group', 'score'])
    first_max_df[first_max_df.columns[0]] = x_list
    first_max_df[first_max_df.columns[1]] = y_list

    # Find the quartiles and IQR for each category
    groups = first_max_df.groupby('group')
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    # Find the outliers for each category
    def outliers(group):
        cat = group.name
        return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']

    out = groups.apply(outliers).dropna()

    # Prepare outlier data for plotting, we need coordinates for every outlier.
    if not out.empty:
        outx = []
        outy = []
        for cat in x_categories_unique_sorted_list:
            # Only add outliers if they exist
            if not out.loc[cat].empty:
                for value in out[cat]:
                    outx.append(cat)
                    outy.append(value)

    p = figure(title=title, plot_width=400, plot_height=350, x_range=x_cat_unique_display_order_list,
               x_axis_label=x_axis_label, y_axis_label=y_axis_label)
    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.score = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, 'score']), upper.score)]
    lower.score = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, 'score']), lower.score)]

    # stems
    p.segment(x_categories_unique_sorted_list, upper.score, x_categories_unique_sorted_list, q3.score, line_color="black")
    p.segment(x_categories_unique_sorted_list, lower.score, x_categories_unique_sorted_list, q1.score, line_color="black")

    # boxes
    p.vbar(x_categories_unique_sorted_list, 0.7, q2.score, q3.score, fill_color="#E08E79", line_color="black")
    p.vbar(x_categories_unique_sorted_list, 0.7, q1.score, q2.score, fill_color="#3B8686", line_color="black")

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(x_categories_unique_sorted_list, lower.score, 0.2, 0.01, line_color="black")
    p.rect(x_categories_unique_sorted_list, upper.score, 0.2, 0.01, line_color="black")

    # outliers
    if not out.empty:
        p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

    p.x(x_list, y_list, color='black')

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"
    p.grid.grid_line_width = 2
    p.xaxis.major_label_orientation = 3.0 / 4

    return p


def get_first_length_count_df(first_list, length_list):
    """
    This function counts how often a certain combinations of (start week of wave, wave length in weeks) occurs and
    returns a data frame containing this information.

    :param first_list: A list, containing the first week of a waves.
    :param length_list: A list, containing the wave lengths.
    :return: A pandas.DataFrame, containing the count of how often a certain combinations of (start week of wave, wave
    length in weeks) occurred
    """
    count_dict = defaultdict(lambda: 0)
    for index in range(len(first_list)):
        count_dict[(first_list[index][1], length_list[index])] += 1
    return pd.DataFrame([[item[0][0], item[0][1], item[1]] for item in count_dict.items()], columns=['first_week',
                                                                                                     'wave_length',
                                                                                                     'count'])


def visualize_data_per_state(input_df, state_str="Baden-Wuerttemberg"):
    """
    This function returns a bokeh figure visualizing the influenza, trend, temperature features for the specified state.

    :param input_df: A pandas.DataFrame, containing rows with names 'state', 'year_week', influenza_week-1,
    influenza_germany_week-1, trend_week-1, trend_germany_week-1, temp_mean-1.
    :param state_str: A str, specifying the state.
    :return: A bokeh figure, visualizing the features for the specified state.
    """

    # Only the rows associated to the current state are considered.
    state_indices = input_df['state'] == state_str
    state_df = input_df[state_indices]

    influenza_list = state_df['influenza_week-1'].tolist()[:-1]
    influenza_germany_list = state_df['influenza_germany_week-1'].tolist()[:-1]
    google_trends_list = state_df['trend_week-1'].tolist()[:-1]
    google_trends_germany_list = state_df['trend_germany_week-1'][:-1]
    temp_list = state_df['temp_mean-1'].div(10).tolist()[:-1]
    # humid_list = state_df['humid_mean-1'].tolist()[:-1]
    # prec_list = state_df['prec_mean-1'].tolist()[:-1]
    dates_list = state_df['year_week'].unique().tolist()[1:]

    p = figure(plot_width=800, plot_height=500, title=state_str, x_axis_label='Date')

    plot_x = [Week(year_week[0], year_week[1]).monday() for year_week in dates_list]

    p.xaxis.formatter = DatetimeTickFormatter(
        years=["%D %B %Y"]
    )

    p.xaxis.major_label_orientation = 3.0 / 4
    p.xaxis[0].ticker.desired_num_ticks = 30

    # Influenza Numbers for Current State and for Germany as a Whole
    p.yaxis.axis_label = '# Influenza Infections'
    p.y_range = Range1d(start=0, end=max(max(influenza_list), max(influenza_germany_list))+3)

    # Google Trends Data
    p.extra_y_ranges['trends'] = Range1d(start=0, end=max(max(google_trends_list), max(google_trends_germany_list)))
    p.add_layout(LinearAxis(y_range_name='trends', axis_label='Google Trends Score'), 'left')

    # Temperature
    p.extra_y_ranges['temp'] = Range1d(start=min(temp_list)-1, end=max(temp_list)+2)
    p.add_layout(LinearAxis(y_range_name='temp', axis_label='Temperature in Degrees Celsius'), 'right')

    # # Precipitation
    # p.extra_y_ranges['prec'] = Range1d(start=0, end=500)
    # p.add_layout(LinearAxis(y_range_name='prec', axis_label='Prec in'), 'right')

    keys_list = ['Influenza', 'Influenza Germany', 'Google Trends', 'Google Trends Germany', 'Temperature']
    argument_list = [influenza_list, influenza_germany_list, google_trends_list, google_trends_germany_list, temp_list]
    color_list = ['#000000', '#5b5b5b', '#1a6864', '#2a9e98', '#c11515']
    y_range_list = ['dummy', 'dummy', 'trends', 'trends', 'temp']
    alpha_list = [1.0, 1.0, 1.0, 1.0, 0.0]
    muted_alpha_list = [0.0, 0.0, 0.0, 0.0, 1.0]

    for index in range(0, 2):
        p.step(plot_x, argument_list[index], legend=keys_list[index], color=color_list[index], alpha=alpha_list[index],
               muted_color=color_list[index], muted_alpha=muted_alpha_list[index])

    for index in range(2, 5):
        p.step(plot_x, argument_list[index], legend=keys_list[index], color=color_list[index], alpha=alpha_list[index],
               muted_color=color_list[index], muted_alpha=muted_alpha_list[index], y_range_name=y_range_list[index])

    p.legend.location = "top_left"
    p.legend.click_policy = "mute"
    p.legend.background_fill_alpha = 0.0

    return p


if __name__ == "__main__":

    # # For Debugging:
    # app.run(debug=True)

    app.run(port=33507)