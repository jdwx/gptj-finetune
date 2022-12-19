#!/bin/sh

# Making sure Python 3.9 and its venv module are installed on your system,
# and finding the correct path to the python 3.9 binary are on you! :-)
# On Ubuntu 20.04 LTS, you need python-3.9, python3.9-venv, and python3.9-dev.
# Or, better yet, build the latest Python 3.9 from source like a boss!
python3.9 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
