from collections import OrderedDict
from isoweek import Week

import pandas as pd
import numpy as np

import dill as pickle

from sklearn.neighbors import KernelDensity

# Plotting
from bokeh.plotting import figure
from bokeh.models import DatetimeTickFormatter, ColumnDataSource, LinearColorMapper, HoverTool
from bokeh.palettes import Category20b
from bokeh.transform import linear_cmap
from bokeh.layouts import column, row
from bokeh.embed import components

from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_moment import Moment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'd90Fj238A679bn940sn4Ghrq9b08a962Nvfm2390'

bootstrap = Bootstrap(app)
moment = Moment(app)

# Getting the data frame containing the features and target variable for all sixteen states from 2005 until 2015.
with open(r'static\Data\datadf.pkl', 'rb') as file:
    data_df = pickle.load(file)



@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/InfluenzaProject', methods=['GET'])
def render_influenza_project():
    # Generate plots
    p_vanilla_influenza = visualize_state_commonalities(data_df)
    p_overall_reported_cases = visualize_overall_reported_cases(data_df)
    p_wave_stats = visualize_wave_stats_distributions(data_df)

    # Embed plots into HTML via Flask Render
    script_vanilla, div_vanilla = components(p_vanilla_influenza)
    script_overall_cases, div_overall_cases = components(p_overall_reported_cases)
    script_wave_stats, div_wave_stats = components(p_wave_stats)

    return render_template('InfluenzaProject.html', script_vanilla=script_vanilla, div_vanilla=div_vanilla,
                           script_overall_cases=script_overall_cases, div_overall_cases=div_overall_cases,
                           script_wave_stats=script_wave_stats, div_wave_stats=div_wave_stats)


#######################
# Visualization methods
#######################

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
                   plot_width=480, plot_height=480,)


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


if __name__ == "__main__":
    app.run(debug=True)