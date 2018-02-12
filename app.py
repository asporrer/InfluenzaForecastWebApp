import os
from collections import OrderedDict, defaultdict
from isoweek import Week

import pandas as pd
import numpy as np

import dill as pickle

from sklearn.neighbors import KernelDensity

# Plotting
from bokeh.plotting import figure
from bokeh.models import Range1d, LinearAxis, DatetimeTickFormatter, ColumnDataSource, LinearColorMapper, \
    LogColorMapper, LabelSet, HoverTool
from bokeh.palettes import Category20b
from bokeh.transform import linear_cmap
from bokeh.layouts import column, row
from bokeh.embed import components
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, log_loss,\
    roc_auc_score, confusion_matrix



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

# Getting the data frame containing the features and target variable for all sixteen states from 2005 until 2015.
with open(data_location, 'rb') as file:
    data_df = pickle.load(file)

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
    return render_template('InfluenzaProject1.html')


@app.route('/InfluenzaProject1', methods=['GET'])
def render_influenza_project1():
    return render_template('InfluenzaProject1.html')


@app.route('/InfluenzaProject2', methods=['GET'])
def render_influenza_project2():

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
    return render_template('contact.html')


@app.route('/waveStatisticsFigure')
def wave_statistics_figure():
    """
    This function is called via an ajax. The code is located in ajaxscripts.js.

    :return: A str, containing the respective html and javascript code.
    """
    # text = request.args.get('jsdata')
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
            # Loading pregenerated plots to speed up loading of the page.
            with open(metric_path_list[index_metric], 'rb') as file:
                script, div = pickle.load(file)

    return render_template('multiselectUpdatedFigure.html', script=script, div=div)


#######################
# Visualization methods
#######################

def generate_new_plots():
    """
    This function generates new versions of the bokeh plots
    transforms them into an embeddable version and than stores them
    to decrease the loading time of the site.

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
    This function returns a bokeh visualization of the influenza waves in the states of Germany from 2005 till 2015.

    :param data_df: A pandas.DataFrame, containing the influenza progression of the sixteen states of Germany.
    :return: A bokeh figure, visualizing the influenza progressions of the sixteen states of Germany.
    """

    kwargs_state_influenza_dict = OrderedDict()

    for current_state in data_df['state'].unique():

            # State information
            state_indices = data_df['state'] == current_state
            state_df = data_df[state_indices]

            kwargs_state_influenza_dict[current_state] = state_df['influenza_week-1'].tolist()[:-1]

    return plot_results('States of Germany', 'Date', '# Influenza Infections per 100.000 Inhabitants',
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


def visualize_overall_reported_cases(data_df):
    """
    This function visualizes the cumulative number of reported influenza infections for each of the sixteen
    states of Germany from 2005 till 2015. The period from the 25th week of 2009 till the 24th week of 2010 is excluded.
    In this period the "outlayer wave" occured (the swine flu).

    :param data_df: A pandas.DataFrame, containing a row with names 'state', 'influenza_week-1'.
    :return: A bokeh figure, visualizing the sum of the overall reported influenza infections for each of the sixteen
    states of Germany.
    """
    input_list = get_number_of_reported_infected_per_state(data_df)

    states = [state_sum[0] for state_sum in input_list]
    sum_of_rep_cases = [state_sum[1] for state_sum in input_list]

    source = ColumnDataSource(data=dict(states=states, sum_of_rep_cases=sum_of_rep_cases))

    TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"

    p = figure(plot_height=500, plot_width=500, x_range=states, y_range=(0, 900), tools=TOOLS,
               title="Total Number of Reported Influenza Infections from 2005-2015",
               y_axis_label='# Reported Influenza Infections per 100 000 inhabitants')

    p.vbar(x='states', top='sum_of_rep_cases', width=0.9, source=source, legend=False,
           line_color='white', fill_color=linear_cmap(field_name='sum_of_rep_cases', palette=["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"], low=0,
                                                      high=900))
    p.line(x=[0, 16], y=[500, 500], color='black')
    p.line(x=[0, 16], y=[300, 300], color='black')

    hover = p.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [("State", "@states"), ("Total Sum of Reported Infections", "@sum_of_rep_cases")]

    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 3.0 / 4
    p.ygrid.grid_line_color = None

    return p


def get_number_of_reported_infected_per_state(input_df):
    """
    This function extracts the overall number of reported influenza infections for each state in Germany and returns
    the respective dictionary. The period from the 25th week of 2009 till the 24th week of 2010 is excluded. In this
    period the "outlayer wave" occured (the swine flu).

    :param input_df: A pandas.DataFrame, containing a row with names 'state', 'influenza_week-1'.
    :return: A dict, holding the sixteen states of Germany as key. The values is the overall sum of reported influenza
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

    first_list = []
    last_list = []
    max_list = []

    # Extracting the wave start, end and height from the data frame.
    for state in input_df['state'].unique().tolist():
        if state in current_states or current_states[0] == 'all':
            helper_first_list, helper_last_list, helper_max_list = get_first_last_max_per_state(input_df, state)
            first_list.extend((helper_first_list))
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



def get_first_last_max_per_state(input_df, state_str, start_year=2005, end_year=2015, start_week=25, end_week=24, target_column_name='influenza_week-1'):
    """

    :param input_df: A pandas.DataFrame,
    :param state_str: A str,
    :return:
    """

    if not ('year_week' in list(input_df.columns) and target_column_name in list(input_df.columns)):
        raise ValueError('The input data frame has to have the column names "year_week" and ' + target_column_name + '.')

    first_list = []
    last_list = []
    max_list = []

    state_df = input_df[input_df['state'] == state_str]
    for year in range(start_year, end_year):
        dummy_df, current_year_df = get_wave_complement_interval_split(state_df, start_year=year, start_week=start_week,
                                                                       end_year=year+1, end_week=end_week)

        max_list.append(current_year_df[target_column_name].max())

        influenza_list = current_year_df[target_column_name].tolist()
        first_index = get_first_or_last_greater_than(influenza_list, 2.0)
        last_index = get_first_or_last_greater_than(influenza_list, 2.0, first=False)



        if first_index is not None:
            helper_first = current_year_df['year_week'].iloc[first_index]
            helper_last = current_year_df['year_week'].iloc[last_index]

            # # Test Code Start:
            # print('year')
            # print(year)
            # print('State')
            # print(state_str)
            # print('first year_week')
            # print(helper_first)
            # print('last year_week')
            # print(helper_last)
            # print('first indices')
            # print(current_year_df[target_column_name].iloc[first_index-1:first_index+2])
            # print('last indices')
            # print(current_year_df[target_column_name].iloc[last_index - 1:last_index + 2])
            # print('')
            # # Test Code end.

            first_list.append((helper_first[0], helper_first[1] - 1))
            last_list.append((helper_last[0], helper_last[1] - 1))
        else:
            first_list.append((year, -1))
            last_list.append((year, -1))

    return (first_list, last_list, max_list)


# Unit test passed
def get_wave_complement_interval_split(data_df, start_year, start_week, end_year, end_week):
    interval_bool_series = data_df['year_week'].apply(
        lambda x: (start_year <= x[0] and (start_week <= x[1] or start_year < x[0])) and (
                x[0] <= end_year and (x[1] <= end_week or x[0] < end_year)))

    interval_complement_bool_series = interval_bool_series.apply(
        lambda x: not x)  # map( lambda x: not x, interval_bool_series )

    interval_df = data_df[interval_bool_series]
    interval_complement_df = data_df[interval_complement_bool_series]

    return interval_complement_df, interval_df


def get_first_or_last_greater_than(input_list, threshold_float, first=True):
    my_range = range(len(input_list))
    if not first:
        my_range = reversed(my_range)
    for index in my_range:
        if 2.0 < input_list[index]:
            return index
    return None


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def visualize_wave_start_vs_severity_via_box(input_df, states=['all']):

    first_last_max_length_lists = list(get_first_last_week_max_length_of_wave(input_df, current_states=states).items())
    first_list = first_last_max_length_lists[0][1]

    week_list = ["Week " + str(year_week[1]) for year_week in first_list]
    max_value_of_wave_list = first_last_max_length_lists[2][1]

    week_categories_unique_list = sorted(list(set(week_list)))
    week_categories_display_order_list = sorted(week_categories_unique_list, key=lambda x: int(x[-2:].strip()))

    return box_plot(week_list, max_value_of_wave_list, week_categories_display_order_list, title="Wave Start vs Wave Severity", x_axis_label="Calender Week", y_axis_label='Number of Infected per 100 000 Inhabitants')


def visualize_wave_start_vs_length_via_box(input_df, states=['all']):
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

    box_fig = box_plot(first_week_list, wave_length_list, week_categories_display_order_list, title="Wave Start vs Wave Length in Weeks", x_axis_label="Calender Week", y_axis_label="Calender Week")
    # Encoding the count of the start week, wave length pair in the x size.
    box_fig.x(x_count_list, y_count_list, color='black', size=np.array(count_for_size_list)*3)

    return box_fig


def box_plot(x_list, y_list, x_cat_unique_display_order_list, title="", x_axis_label="", y_axis_label=""):

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

    p = figure(title=title, plot_width=400, plot_height=350, x_range=x_cat_unique_display_order_list, x_axis_label=x_axis_label,
               y_axis_label=y_axis_label)
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
    #p.xaxis.major_label_text_font_size = "12pt"
    p.xaxis.major_label_orientation = 3.0 / 4

    return p


def get_first_length_count_df(first_list, length_list):
    count_dict = defaultdict(lambda: 0)
    for index in range(len(first_list)):
        count_dict[(first_list[index][1], length_list[index])] += 1
    return pd.DataFrame([[item[0][0], item[0][1], item[1]] for item in count_dict.items()], columns=['first_week',
                                                                                                     'wave_length', 'count'])


def visualize_data_per_state(input_df, state_str="Baden-Wuerttemberg"):

    # State information
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

    plot_x = [Week(year_week[0], year_week[1]).monday() for year_week in
                  dates_list]

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

###
# Start: Metric Figures
###

def visual_metrics():

    # Getting the instance variables.
    with open(data_for_metric_location, 'rb') as file:
        week_threshold_years_list = pickle.load(file)

    proba_threshold_list = [0.5, 0.4]

    metric_list = [accuracy_score, precision_score, recall_score, lambda x, y: fbeta_score(x, y, 2.0), log_loss,
                   roc_auc_score]
    needs_binary_pred_bool_list = [True, True, True, True, False, False]
    metric_needs_one_actual_bool_list = [False, False, True, True, True, True]
    metric_name_list = ['Accuracy', 'Precision', 'Recall', 'F2 Score', 'Log Loss', 'AUC']
    title_metric_plot_list = [metric + " per Test Year" for metric in metric_name_list]

    threshold_list = [0.8, 7.0]
    threshold_color_list = ["#036564", "#550b1d"]

    metric_plot_list = []
    metricPlotLists = [[], ([], [])]

    for index in range(len(metric_list)):
        metric_dict = get_metric_for_week_threshold(week_threshold_years_list, metric=metric_list[index],
                                                    metric_needs_binary_predictions_bool=
                                                    needs_binary_pred_bool_list[index],
                                                    metric_needs_one_actual_bool=metric_needs_one_actual_bool_list[
                                                        index],
                                                    proba_threshold1=proba_threshold_list[0],
                                                    proba_threshold2=proba_threshold_list[1])

        if index == 0:

            for index_threshold in range(2):
                plot_list = []
                for index_week in range(len(metric_dict['sorted_unique_week_list'])):
                    plot_list.append(plot_confusion_matrix(
                        metric_dict['all_true_values_ndarray_per_week_threshold_tuple'][index_threshold][index_week],
                        metric_dict['all_predictions_ndarray_per_week_threshold_tuple'][index_threshold][index_week],
                        title='Week ' + str(index_week), threshold_proba=proba_threshold_list[index_threshold]))
                metricPlotLists[1][index_threshold].extend(plot_list)
                time.sleep(2)

        metric_plot_list.append(
            visualize_scores(metric_dict['week_str_list'], metric_dict['metric_value_list'], metric_dict['year_list'],
                             metric_dict['threshold_list'],
                             metric_dict['overall_metric_per_week_and_threshold_tuple'][0],
                             metric_dict['overall_metric_per_week_and_threshold_tuple'][1],
                             metric_dict['sorted_unique_week_list'], threshold1=threshold_list[0],
                             threshold2=threshold_list[1], title=title_metric_plot_list[index],
                             y_axis_label=metric_name_list[index], metric_name=metric_name_list[index],
                             color_threshold1=threshold_color_list[0],
                             color_threshold2=threshold_color_list[1]))

    metricPlotLists[0].extend(metric_plot_list)

    with open(os.path.join(os.path.dirname(__file__), r'Data\Plots\metricPlotLists.pkl'), 'wb') as file:
        pickle.dump(metricPlotLists, file)

    with open(os.path.join(os.path.dirname(__file__), r'Data\Plots\metricPlotLists.pkl'), 'rb') as file:
        metricPlotLists = pickle.load(file)

    show(row(metricPlotLists[0]))
    time.sleep(2)
    show(row(metricPlotLists[1][0]))
    time.sleep(2)
    show(row(metricPlotLists[1][1]))
    time.sleep(2)

    return None

def visualize_confusion_matrices():
    return None

def get_metric_for_week_threshold(week_threshold_years_list, metric=roc_auc_score, proba_threshold1=0.5, proba_threshold2=0.5,
                                  metric_needs_one_actual_bool=True, metric_needs_binary_predictions_bool=True):
    """
    Possible metrics are: auc, log_loss, accuracy_score, precision, recall, f_score

    :param metric:
    :return:
    """

    week_list = []
    threshold_list = []
    year_list = []

    actual_values_list = []
    predictions_list = []

    metric_value_list = []
    has_ones_year_bool_list = []

    threshold1 = 0.8
    threshold2 = 7.0

    for per_week_dict in week_threshold_years_list:
        for per_threshold_dict in per_week_dict['output_by_threshold']:

            if per_threshold_dict['threshold'] != threshold1 and per_threshold_dict['threshold'] != threshold2:
                raise ValueError('A threshold of the loaded file has an unexpected value.')

            current_year = 2005
            for pred_proba_true_tuple in per_threshold_dict['output_per_year']:

                if current_year != 2005:

                    if pred_proba_true_tuple[1].sum() != 0:
                        has_ones_year_bool_list.append(True)
                    else:
                        has_ones_year_bool_list.append(False)

                    week_list.append(per_week_dict['week'])  # week'] + 1 depends on week_threshold_years_list format
                    threshold_list.append(per_threshold_dict['threshold'])
                    year_list.append(current_year)
                    actual_values_list.append(pred_proba_true_tuple[1].flatten())
                    predictions_list.append(pred_proba_true_tuple[0])

                    current_correct_format_predictions_ndarray = pred_proba_true_tuple[0]

                    if metric_needs_binary_predictions_bool:
                        if per_threshold_dict['threshold'] == threshold1:
                            current_proba_threshold = proba_threshold1
                        else:
                            current_proba_threshold = proba_threshold2

                        current_correct_format_predictions_ndarray = \
                            (current_proba_threshold <= current_correct_format_predictions_ndarray).astype(int)

                    if pred_proba_true_tuple[1].sum() != 0 or not metric_needs_one_actual_bool:
                        metric_value_list.append(
                            metric(pred_proba_true_tuple[1], current_correct_format_predictions_ndarray))
                    else:
                        metric_value_list.append(None)

                if current_year != 2008:
                    current_year += 1
                else:
                    current_year += 2

    # The unique list of weeks for the x_range of the figure.
    sorted_unique_week_list = sorted(list(set(week_list)))
    sorted_unique_week_list = ['Week ' + str(week) for week in sorted_unique_week_list]

    # Concatenating the true and predicted values per threshold and week.

    all_predictions_ndarray_per_week_threshold1 = get_true_and_predicted_values_per_week(predictions_list,
                                                                                         threshold_list, week_list,
                                                                                         sorted_unique_week_list,
                                                                                         threshold1)
    all_true_values_ndarray_per_week_threshold1 = get_true_and_predicted_values_per_week(actual_values_list,
                                                                                         threshold_list, week_list,
                                                                                         sorted_unique_week_list,
                                                                                         threshold1)
    all_predictions_ndarray_per_week_threshold2 = get_true_and_predicted_values_per_week(predictions_list,
                                                                                         threshold_list,
                                                                                         week_list,
                                                                                         sorted_unique_week_list,
                                                                                         threshold2)
    all_true_values_ndarray_per_week_threshold2 = get_true_and_predicted_values_per_week(actual_values_list,
                                                                                         threshold_list, week_list,
                                                                                         sorted_unique_week_list,
                                                                                         threshold2)

    # Calculating the scores per threshold and week.
    score_per_week_threhold1 = []
    score_per_week_threhold2 = []

    for index in range(len(sorted_unique_week_list)):
        formatted_predictions1_ndarray = all_predictions_ndarray_per_week_threshold1[index]
        formatted_predictions2_ndarray = all_predictions_ndarray_per_week_threshold2[index]
        if metric_needs_binary_predictions_bool:
            formatted_predictions1_ndarray = (proba_threshold1 <= formatted_predictions1_ndarray).astype(int)
            formatted_predictions2_ndarray = (proba_threshold2 <= formatted_predictions2_ndarray).astype(int)
        score_per_week_threhold1.append(metric(all_true_values_ndarray_per_week_threshold1[index],
                                               formatted_predictions1_ndarray))
        score_per_week_threhold2.append(metric(all_true_values_ndarray_per_week_threshold2[index],
                                               formatted_predictions2_ndarray))

    # The str week list for plotting the x_axis.
    week_str_list = ['Week ' + str(week) for week in week_list]

    if metric_needs_one_actual_bool:
        # Plotting the different AUC values for each week, threshold and test year.
        week_str_list = np.array(week_str_list)[has_ones_year_bool_list]
        metric_value_list = np.array(metric_value_list)[has_ones_year_bool_list]
        year_list = np.array(year_list)[has_ones_year_bool_list]
        threshold_list = np.array(threshold_list)[has_ones_year_bool_list]

    # Calculating the overall metric scores for the different years.
    overall_metric_per_week_and_threshold1 = score_per_week_threhold1

    # At the moment not used
    # get_average_auc_per_week(auc_list, auc_threshold_list, auc_week_list,
    # sorted_unique_week_list,
    # 0.8)

    overall_metric_per_week_and_threshold2 = score_per_week_threhold2

    # At the moment not used
    # get_average_auc_per_week(auc_list, auc_threshold_list, auc_week_list,
    # sorted_unique_week_list,
    # 7.0)

    return {'week_str_list': week_str_list, 'metric_value_list': metric_value_list, 'year_list': year_list,
            'threshold_list': threshold_list, 'overall_metric_per_week_and_threshold_tuple':
                (overall_metric_per_week_and_threshold1, overall_metric_per_week_and_threshold2),
            'sorted_unique_week_list': sorted_unique_week_list, 'all_true_values_ndarray_per_week_threshold_tuple':
                (all_true_values_ndarray_per_week_threshold1, all_true_values_ndarray_per_week_threshold2),
            'all_predictions_ndarray_per_week_threshold_tuple': (all_predictions_ndarray_per_week_threshold1,
                                                                 all_predictions_ndarray_per_week_threshold2)}


def get_true_and_predicted_values_per_week(value_list, threshold_list, week_list, sorted_unique_week_list, threshold):

    value_ndarray_per_week_list = [None for _ in sorted_unique_week_list]

    for index in range(len(threshold_list)):
        if threshold_list[index] == threshold:
            if value_ndarray_per_week_list[week_list[index]-1] is None:
                value_ndarray_per_week_list[week_list[index]-1] = value_list[index]
            else:
                value_ndarray_per_week_list[week_list[index]-1] = np.hstack([value_ndarray_per_week_list[week_list[index]-1], value_list[index]])

    return value_ndarray_per_week_list


def visualize_scores(week_str_list, metric_value_list, year_list, threshold_list,
                     overall_metric_per_week_and_threshold1, overall_metric_per_week_and_threshold2,
                     sorted_unique_week_list, threshold1=0.8, threshold2=7.0, title='', y_axis_label='Metric Value',
                     color_threshold1="#036564", color_threshold2="#550b1d", metric_name='Metric'):

    TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"

    p = figure(title=title, x_range=sorted_unique_week_list, x_axis_label='Forecasting Distance',
               y_axis_label=y_axis_label, tools=TOOLS)

    # Plotting metric data per year for two thresholds.
    index_bool_threshold1_ndarray = np.array(threshold_list) == threshold1
    index_bool_threshold2_ndarray = np.logical_not(index_bool_threshold1_ndarray)

    index_list = [index_bool_threshold1_ndarray, index_bool_threshold2_ndarray]
    color_list = [color_threshold1, color_threshold2]
    thresh_list = [threshold1, threshold2]

    for index in range(2):
        week_str_ndarray = np.array(week_str_list)[index_list[index]]

        metric_value_ndarray = np.array(metric_value_list)[index_list[index]]

        year_ndarray = np.array(year_list)[index_list[index]]

        threshold_ndarray = np.array(threshold_list)[index_list[index]]

        metric_data_dict = dict(x=week_str_ndarray, y=metric_value_ndarray, year=year_ndarray,
                                threshold=threshold_ndarray)

        metric_source = ColumnDataSource(data=metric_data_dict)

        p.x(x='x', y='y', size=10, color=color_list[index], source=metric_source,
            legend='Per Year ' + metric_name + ': Threshold ' + str(thresh_list[index]), muted_color=color_list[index],
            muted_alpha=0.0)

    # Plotting the Average over the AUC values for the different years.
    average1_auc_source = ColumnDataSource(data=dict(x=sorted_unique_week_list, y=overall_metric_per_week_and_threshold1,
                                                     year=['2005, ..., 2008, 2010, ... 2014'] * len(sorted_unique_week_list),
                                                     threshold=['0.8'] * len(sorted_unique_week_list)))
    average2_auc_source = ColumnDataSource(data=dict(x=sorted_unique_week_list, y=overall_metric_per_week_and_threshold2,
                                                     year=['2005, ..., 2008, 2010, ... 2014'] * len(
                                                         sorted_unique_week_list),
                                                     threshold=['7.0'] * len(sorted_unique_week_list)))
    p.rect(x='x', y='y', width=0.8, height=5, source=average1_auc_source, color=color_threshold1,
           muted_color=color_threshold1, muted_alpha=0.0, legend=metric_name + ' Overall: Threshold ' + str(threshold1),
           height_units="screen")
    p.rect(x='x', y='y', width=0.8, height=5, source=average2_auc_source, color=color_threshold2,
           muted_color=color_threshold2, muted_alpha=0.0, legend=metric_name + 'Overall: Threshold ' + str(threshold2),
           height_units="screen")

    # Some formatting and interaction
    p.xaxis.major_label_orientation = 3.0 / 4
    p.legend.location = "bottom_left"
    p.legend.background_fill_alpha = 0.0
    p.legend.click_policy = "mute"

    hover = p.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [("Year(s)", "@year")]

    return p


def plot_confusion_matrix(actual_classes_ndarray, predicted_probabilities_ndarray, threshold_proba=0.05, title=''):

    true_values_ndarray_per_week = actual_classes_ndarray
    predicted_values_ndarray_per_week = (threshold_proba <= predicted_probabilities_ndarray).astype(int)

    cur_confusion_matrix = confusion_matrix(true_values_ndarray_per_week, predicted_values_ndarray_per_week)

    true_negatives = cur_confusion_matrix[0, 0]
    false_positives = cur_confusion_matrix[0, 1]
    false_negatives = cur_confusion_matrix[1, 0]
    true_positives = cur_confusion_matrix[1, 1]

    # Categories for the x and y-axis.
    labels_prediction = ('Pred. Class 0', 'Pred. Class 1')
    labels_truth = ('Class 0', 'Class 1')

    start_length_count_df = pd.DataFrame(
        [[labels_prediction[0], labels_truth[0], true_negatives], [labels_prediction[1], labels_truth[0], false_positives],
         [labels_prediction[0], labels_truth[1], false_negatives],
         [labels_prediction[1], labels_truth[1], true_positives]], columns=['predicted_class', 'true_class', 'count'])

    # Coloring
    colors = ["#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]

    mapper = LogColorMapper(palette=colors, low=0, high=7000)

    source = ColumnDataSource(start_length_count_df)

    p = figure(title=title, x_range=labels_prediction, y_range=list(labels_truth)[::-1],
               plot_width=200, plot_height=200,
               x_axis_location="above")

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "7pt"
    p.axis.major_label_standoff = 0

    p.rect(x='predicted_class', y='true_class', width=1, height=1,
           source=source,
           fill_color={'field': 'count', 'transform': mapper},
           line_color=None)

    p.toolbar.logo = None
    p.toolbar_location = None

    labels = LabelSet(x='predicted_class', y='true_class', text='count', level='glyph',
                      x_offset=0, y_offset=0, source=source, render_mode='canvas', text_align="center",
                      text_baseline="middle", text_color="black")

    p.add_layout(labels)

    # q.yaxis.major_label_orientation = math.pi / 4

    return p


###
# End: Metric Figures
###


if __name__ == "__main__":
     app.run(debug=True)
    # app.run(port=33507)