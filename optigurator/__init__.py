from flask import Flask, render_template

from optigurator.views import bp
from optigurator.onshape import onshape


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.secret_key = "dev"

    app.register_blueprint(onshape, url_prefix="/oauth")
    app.register_blueprint(bp)

    return app
