from collections import OrderedDict
from isoweek import Week

import dill as pickle

# Plotting
from bokeh.plotting import figure
from bokeh.models import DatetimeTickFormatter
from bokeh.palettes import Category20b
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
    p_vanilla_influenza = visualize_state_commonalities(data_df)

    # Embed plot into HTML via Flask Render
    script, div = components(p_vanilla_influenza)

    return render_template('InfluenzaProject.html', script=script, div=div)


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
               y_axis_label=y_axis_title_param)

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

    return p


if __name__ == "__main__":
    app.run(debug=True)