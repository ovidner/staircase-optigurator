Staircase optigurator
=====================
The Staircase optigurator is an optimizing configurator for spiral staircases. It utilizes [Flask](http://flask.pocoo.org) and [OpenMDAO](http://openmdao.org) in conjunction with [Onshape](https://www.onshape.com) to calculate and present pareto-optimal spiral staircase solutions to case-specific specifications. It was developed as part of a research project between [Linköping University](https://liu.se), [Weland AB](https://www.weland.se) and others.

This is strictly for experimental use.

# Prerequisites
* The application uses the Onshape feature available in https://cad.onshape.com/documents/d69f2c5f691ab246b4955287 as an input. Add the feature to your Onshape account.
* You need to register an OAuth application at the Onshape developer portal to gain access to the Onshape API. See separate instructions.
* To run this under (less un)realistic circumstances, you need some proprietary data in the `optigurator_data` folder. See separate instructions.

# Setting up a development environment and starting the application server
1. Make sure you have Python 3.7 and Pipenv installed on your computer. The procedure differs largely on different platforms.
2. Clone this repository to your computer.
3. If you have received data files from the authors, copy them into the data folder. If not, duplicate all `.example` files in the data folder and remove the `.example` part of their filename. Modify them accordingly.
4. Start a command-line shell in the repository root and run `pipenv install --dev` followed by `pipenv run dev-server`.
5. Navigate your browser to https://localhost:5000. Accept the HTTPS certificate or generate your own key and certificate (replace the `dev-https.(crt|key)` files).

# Registering an Onshape OAuth application
This is not needed if you have received data/settings files from the authors.

1. Go to the [corresponding page at the Onshape developer portal](https://dev-portal.onshape.com/oauthApps/createNew).
2. Fill in the needed information:
    * Name: _Whatever you want_
    * Primary format: _Whatever you want_
    * Summary: _Whatever you want_
    * Redirect URLs:
      * `https://localhost:5000/oauth/onshape/authorized`
      * `https://127.0.0.1:5000/oauth/onshape/authorized`
    * iframe URL: `https://localhost:5000`
    * Supports collaboration: Yes
    * Permissions:
      * Application can read your profile information
      * Application can read your documents
3. Submit and note the client ID and secret. Add them accordingly to the `optigurator_data/settings.py` file.
4. (optional) If you want to use the application from within Onshape (as a document tab/element), you also need to register a store entry for the OAuth application.
