# Freethrow EEG - Record EEG during freethrow shooting

This program finds a MUSE bluetooth device and then starts recording alpha power from it.

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

---

# Plan

KYLE -brainstorm Feb 25

1. Sync with video of freethrows on tripod, video shooter and the basket
2. Add prompts in the terminal to start and stop recording each shot
3. Add prompts to enter success or failure.
4. Start the video synced with EEG
5. Might record all EEG spectra not just alpha incase its interesting.
