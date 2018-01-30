from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_moment import Moment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'd90Fj238A679bn940sn4Ghrq9b08a962Nvfm2390'

bootstrap = Bootstrap(app)
moment = Moment(app)

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
    return render_template('InfluenzaProject.html')


if __name__ == "__main__":
    app.run(debug=True)