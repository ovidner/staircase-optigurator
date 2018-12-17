import os
from functools import partial, wraps

from flask import Blueprint, redirect, render_template, request, url_for
from oauthlib.oauth2 import TokenExpiredError

from optigurator.onshape import onshape
from optigurator.optimization import run_optimization
from optigurator.pareto import generate_pareto_cases
from optigurator.utils import (
    format_degree,
    format_millimeter,
    format_percent,
    problem_constants_from_onshape_feature,
    recording_filename,
    ureg,
)

bp = Blueprint("main", __name__, static_folder="static", template_folder="templates")


def require_onshape_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not onshape.session.authorized:
            return redirect(url_for("onshape.login"))
        try:
            return func(*args, **kwargs)
        except TokenExpiredError:
            return redirect(url_for("onshape.login"))

    return wrapper


@bp.route("/")
@require_onshape_auth
def index():
    # Onshape provides these for us, neato!
    document_id = request.args.get("documentId", None)
    workspace_id = request.args.get("workspaceId", None)
    if document_id and workspace_id:
        return redirect(
            url_for("main.elements", document_id=document_id, workspace_id=workspace_id)
        )

    # TODO: pagination
    return render_template("index.html", data=onshape.session.get("documents").json())


@bp.route("/d/<document_id>/w/<workspace_id>/elements")
@require_onshape_auth
def elements(document_id, workspace_id):
    data = onshape.session.get(
        f"documents/d/{document_id}/w/{workspace_id}/elements"
    ).json()
    return render_template(
        "elements.html",
        data=data,
        url_for_element=partial(
            url_for, "main.features", document_id=document_id, workspace_id=workspace_id
        ),
        document_id=document_id,
        workspace_id=workspace_id,
    )


@bp.route("/d/<document_id>/w/<workspace_id>/e/<element_id>/features")
@require_onshape_auth
def features(document_id, workspace_id, element_id):
    data = onshape.session.get(
        f"partstudios/d/{document_id}/w/{workspace_id}/e/{element_id}/features"
    ).json()
    url_for_optimization = partial(
        url_for,
        "main.optimize",
        document_id=document_id,
        workspace_id=workspace_id,
        element_id=element_id,
    )
    return render_template(
        "features.html",
        data=data,
        url_for_optimization=url_for_optimization,
        document_id=document_id,
        workspace_id=workspace_id,
        element_id=element_id,
    )


@bp.route(
    "/d/<document_id>/w/<workspace_id>/e/<element_id>/f/<feature_id>/optimize",
    methods=["POST"],
)
@require_onshape_auth
def optimize(document_id, workspace_id, element_id, feature_id):
    fs = """
        function(context is Context, queries is Map) {{
            return getVariable(context, "stairDefinition_{feature_id}");
        }}
    """.format(
        feature_id=feature_id
    )

    response = onshape.session.post(
        f"partstudios/d/{document_id}/w/{workspace_id}/e/{element_id}/featurescript",
        json={"script": fs, "queries": []},
    ).json()

    problem_constants = problem_constants_from_onshape_feature(response["result"])
    problem_recording_filename = recording_filename(problem_constants.id)

    if not os.path.isfile(problem_recording_filename):
        run_optimization(problem_constants)

    pareto_cases = list(generate_pareto_cases(problem_constants))

    return render_template(
        "optimize.html",
        problem_constants=problem_constants,
        pareto_cases=pareto_cases,
        format_deg=format_degree,
        format_mm=format_millimeter,
        format_percent=format_percent,
        document_id=document_id,
        workspace_id=workspace_id,
        element_id=element_id,
        feature_id=feature_id,
        recording_filename=problem_recording_filename,
    )
