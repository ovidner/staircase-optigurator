from os import environ

from flask_dance.consumer import OAuth2ConsumerBlueprint

onshape = OAuth2ConsumerBlueprint(
    "onshape",
    __name__,
    base_url="https://cad.onshape.com/api/",
    authorization_url="https://oauth.onshape.com/oauth/authorize",
    token_url="https://oauth.onshape.com/oauth/token",
    client_id=environ.get("ONSHAPE_CLIENT_ID", ""),
    client_secret=environ.get("ONSHAPE_CLIENT_SECRET", ""),
)
