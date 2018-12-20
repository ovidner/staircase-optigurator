import os

from flask import Flask, render_template

from optigurator.views import bp
from optigurator.onshape import onshape


def create_app(test_config=None, data_dir=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config["DATA_DIR"] = data_dir
    app.config.from_pyfile(os.path.join(data_dir, "settings.py"))

    app.register_blueprint(onshape, url_prefix="/oauth")
    app.register_blueprint(bp)

    return app
