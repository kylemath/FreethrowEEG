# Freethrow EEG - Record EEG during freethrow shooting

This program finds a MUSE bluetooth device and then starts recording alpha power from it.

---

Changelog:

> Feb 25 - Kyle - Made Readme file, requirements.txt file, and github repo

---

To install and run locally:

The first time you setup the project you can clone the repo from github and then make a virtual environment to install all the toolboxes to isolate them from other parts of your computer.

Only the first time:
open a terminal window on mac and type:

```bash
git clone http://github.com/kylemath/FreethrowEEG
cd FreethrowEEG
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python test.py
```

Then in the future:
open a terminal:

```bash
cd FreethrowEEG
source venv/bin/activate
python test.py
```
