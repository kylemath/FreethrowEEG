# Chat: Royce Abstract and EEG Analysis Pipeline

*Transcript ID: aef7d98e-3742-4c0c-81dd-181687777df4*

---

## User

@Presentations/Royce2026Abstract.md is an abstract my student is working on. Lets add a title to it and can you check that it properly describes our methods

---

## Assistant

Let me read the abstract file and explore the project to understand the methods being used.

---

## Assistant

Now I have a clear picture of both the abstract and the actual methods. Let me note what I found before editing:

**Accuracy issues in the abstract:**
1. It calls the system "web-based" — it's actually a Python desktop application
2. It omits the core signal processing method: frequency band analysis (delta, theta, alpha, beta, gamma) via Butterworth bandpass filtering
3. It doesn't mention the success/failure comparison, which is a key part of the study design

Let me add a title and address these points:

*[Analysis of abstract vs. actual methods, title consideration...]*

---

## Assistant

Here's a summary of what I changed and why:

**Title added:**
> *Toward Ecological Validity in Sport Neuroscience: Using Portable EEG to Examine Neural Readiness During Basketball Free-Throw Shooting*

This frames the contribution clearly — the ecological validity angle is the central claim, and it signals both the domain (sport neuroscience) and the task (free-throw shooting).

**Three methodological corrections/additions:**

1. **"web-based system" → "desktop-based data collection system"** — The project is a Python desktop GUI application using BrainFlow, not a web app. This was factually inaccurate.

2. **Added channel names and frequency band details** — The abstract previously said nothing about *how* EEG was analyzed. Adding that frequency band power (delta through gamma) was extracted via Butterworth filters in a 1-second sliding window, averaged across 4 channels, accurately reflects the core signal processing pipeline.

3. **Added success/failure comparison** — The study design explicitly compares made vs. missed shots, which is the key dependent variable. The original abstract implied the study only looked for a readiness marker without mentioning that outcome was actually being tested.

---

## User

add authors Lukas Laskowski and Kyle E. Mathewson

---

## Assistant

Done — authors added between the title and the abstract body.

---

## User

Id like to create a python script for anslysis for this data. The script should load data files, for now: @data/freethrow_session_lukas_test_2026-02-16.json and start to test some of our hyopthesis from this pilot data, us I should be able to plot the raw data, add high, low, butterworth filtering, for the raw data stream, and I want to show time graphs of the power in each band over the session.

I want to clearly show where in the continuous data the shots appeared, I want to show the success vs failure of each shot. I want to show averaged power over all shots over the shot period, and then differential success and failture shots for each band power.

Make any other test or analysis. Create a template of a .tex file report and then make the python scripts wrapper populate the tex with new figures and stats, adjust the text to interpret the results, and compile the .tex into a pdf report

---

## Assistant

The user wants me to create a comprehensive analysis pipeline for their EEG free-throw data. Let me break this down:

1. Python analysis script that:
   - Loads the JSON data
   - Plots raw EEG data
   - Applies filtering (high-pass, low-pass, Butterworth)
   - Shows power in each frequency band over time
   - Marks shot locations in continuous data
   - Shows success vs failure for each shot
   - Shows averaged power over all shots during shot period
   - Shows differential success vs failure for each band
   - Additional analyses

2. LaTeX template report

3. Python wrapper that populates the LaTeX with figures and stats, adjusts text, compiles to PDF

I need to first understand the data structure and then build out the implementation step by step.

---

## Assistant

Let me look at more of the data structure to understand the continuous EEG stream and shot boundaries.

Now let me examine the JSON structure more carefully. I need to understand:
1. The structure of the shots array (each shot has prep, shot, post phases with band data)
2. The continuous EEG data structure
3. The raw data stream structure

---

## Assistant

Now I understand the data structure:

**Top-level:**
- `playerName`: string
- `timestamp`: ISO datetime
- `totalShots`: int (10)
- `shots`: array of shot objects
- `mode`: "real"
- `eegData`: continuous EEG data with timestamps and band powers
- `sessionDuration`: float
- `completedAt`: ISO datetime

**Each shot:**
- `shotNumber`: int
- `timestamp`: ISO datetime
- `success`: boolean
- `duration`: float (cumulative time from session start)
- `eegData`: object with phases (prep, shot, post) and band arrays (delta, theta, alpha, beta, gamma)

**Each band data point:**
- `timestamp`: float (relative seconds from session start)
- `power`: float (band power)
- `raw`: float (raw signal value)

---

## Assistant

Now I understand the full structure. The continuous EEG data is in `eegData.timestamps` (array of float), and `eegData.bands.delta`, `eegData.bands.theta`, etc. are arrays of power values corresponding to those timestamps.

---

## Assistant

Now I have a complete picture of the data. Let me build the full analysis pipeline.

**Continuous EEG data:**
- `eegData.timestamps`: 1392 samples, ~0.25s apart (4 Hz), from 0.255s to 347.999s
- `eegData.bands.{delta,theta,alpha,beta,gamma}`: power values (1392 each)
- `eegData.rawBands.{delta,theta,alpha,beta,gamma}`: raw filtered values (1392 each)

**Shot data (10 shots):**
- Each shot has 5 phases: prep, preShot, recording, postShot, review
- Each phase contains 5 bands with arrays of {timestamp, power, raw} values
- 3 successful shots (6, 9, 10) and 7 failures

---

## Assistant

Now let me create the wrapper script that runs the analysis, populates the template, and compiles the PDF.

---

## Assistant

Good, both dependencies installed and pdflatex is available. Now let me run the full pipeline.

---

## Assistant

Everything worked end-to-end. Let me verify the outputs and check the stats.

---

## Assistant

The full pipeline ran successfully and produced the report. Here's what was created:

---

**Files created in `analysis/`:**

| File | Purpose |
|------|---------|
| `analyze_session.py` | Core analysis: loads JSON, generates 8 figures, computes stats |
| `report_template.tex` | LaTeX template with placeholders for all figures, stats, and interpretive text |
| `generate_report.py` | Wrapper: runs analysis, fills template with data-driven text, compiles PDF |
| `report.pdf` | The compiled 388 KB report (ready to open) |
| `session_stats.json` | All computed statistics in structured JSON |
| `figures/` | 8 publication-quality PDF figures |
| `venv/` | Virtual environment with numpy, scipy, matplotlib |

**The 8 figures:**

1. **Continuous band power** -- all 5 bands over the full session with shot markers (made=green, missed=red)
2. **Filtering demonstration** -- raw delta signal with low-pass, high-pass, and bandpass Butterworth filters
3. **Shot-locked averages** -- ERP-style time-locked averages aligned to execution onset (all shots)
4. **Success vs failure** -- shot-locked comparison between made (n=3) and missed (n=7)
5. **Phase bar chart** -- mean power per phase (pre-shot, execution, post-shot) by outcome, per band
6. **Theta/alpha ratio** -- continuous ratio + boxplot by outcome
7. **Shot progression** -- pre-shot power across 10 shots with linear trend lines
8. **Session raster** -- timeline overview with shaded trial phases

**Statistical highlights from this pilot (3/10 made):**
- Theta showed the largest effect size for pre-shot power (Cohen's *d* = 0.64), with made shots having slightly higher pre-shot theta
- Alpha and beta showed medium effect sizes (*d* ~ -0.54 and -0.58), both slightly lower for made shots
- No comparisons reached significance (expected with n=3 vs n=7)
- Theta/alpha ratio was marginally higher for made (1.68) vs missed (1.67)

**To re-run the pipeline on any session:**

```bash
source analysis/venv/bin/activate
python analysis/generate_report.py data/freethrow_session_lukas_test_2026-02-16.json
```

The report text auto-adapts to the data -- all interpretive paragraphs, the discussion, table, and conclusion are generated from the actual statistics.
