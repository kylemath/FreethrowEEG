## Determining Trial Counts for a FreethrowEEG Pilot

**Participants:**
- **Student (S)** – Neuroscience master's student running a pilot with FreethrowEEG + Muse
- **Expert (E)** – PhD EEG researcher specializing in consumer EEG (especially Muse)

---

### 1. Setting the Goal

**S:** I'm planning a single-participant pilot using the FreethrowEEG program with a Muse headband. I want to know how many free throw trials I should collect. Where do I even start?  
**E:** Good question. Before we talk numbers, we need to clarify what kind of signal you care about and how you plan to analyze it. Are you thinking more about **ERP-style, time-locked responses** or **spectral/oscillatory power changes** around the free throw?

**S:** Mostly spectral changes (alpha, beta) around the shot, but it would be nice if the data could also support some simple ERP-style averages if possible.  
**E:** Great, so we need a design that is **reasonable for spectral analyses** but not obviously underpowered if you later peek at ERPs. Let’s first anchor on typical trial counts from lab-grade EEG, then adjust for Muse and movement.

---

### 2. Typical Trial Counts in EEG Experiments

**S:** What’s “typical” in standard EEG work?  
**E:** It depends on the paradigm, but ballpark:
- **Simple ERP paradigms (e.g., visual oddball):** often **60–120 artifact-free trials per condition**, sometimes more.
- **Cognitive/attention ERPs (P3, N2, etc.):** **100–200+ usable trials per condition** is common.
- **Spectral/oscillatory analyses:** can work with **fewer trials** if each trial contains a reasonably long time window (e.g., several seconds), but more trials still help stabilize estimates.

**S:** And that’s after artifact rejection, right?  
**E:** Exactly. Raw collection might be **30–50% higher** than the final “clean” count, depending on the task and participant movement.

---

### 3. Noise and Artifact Issues with Muse / Consumer EEG

**S:** How does this change with a Muse system instead of lab EEG?  
**E:** Muse has a few constraints:
- **Fewer channels** and **dry electrodes**, so the signal is noisier and more sensitive to motion.
- The **forehead/ear locations** can pick up a lot of **muscle activity** (especially facial tension, jaw clenching, and neck/shoulder engagement during shooting).
- The headband may **shift slightly** with repeated shots, adding slow drifts and intermittent contact loss.

**S:** So I should expect more noisy trials than in a standard seated lab task?  
**E:** Yes. In a **stationary, eyes-open Muse recording**, you might retain **70–80%** of trials if the participant behaves nicely. In an **active motor task like basketball shooting**, it’s safer to assume **only ~40–60% of trials are clean enough** for higher-quality analyses, especially for components that are sensitive to muscle artifacts.

---

### 4. Expected Trial Loss from Movement Artifacts

**S:** What kinds of artifacts are we talking about during free throws?  
**E:** Several:
- **Head movement** as the player bends, extends, and follows through.
- **Neck and facial muscle activation** that contaminates especially higher frequencies (beta/gamma).
- **Transient contact changes** when the band shifts with motion or sweat.
- Occasional **blinks and eye movements** tied to looking at the rim, ball, etc.

**S:** How much trial loss should I realistically budget for?  
**E:** For a motor-heavy task with consumer EEG:
- Plan for at least **40–50% of trials being compromised** to some degree.
- If you’re conservative and use strict artifact rejection, you might end up keeping **about 1 in 2** trials, sometimes worse if the fit or environment is poor.

**S:** So if I think I need 60 clean trials, I might have to collect ~120 in total?  
**E:** That’s a reasonable **first-order estimate**, yes.

---

### 5. ERP-Style vs Spectral Analyses

**S:** Given FreethrowEEG, I’ll have segments like prep, pre-shot, recording, post-shot. How do trial counts differ for ERPs vs spectral measures here?  
**E:** Let’s separate them:

- **ERP-style analyses:**
  - ERPs want **many short, consistently time-locked trials**.
  - Movement-related tasks make clean ERPs harder because **muscle and movement artifacts are time-locked to the action too**.
  - For a **pilot with Muse and one participant**, aiming for **40–80 clean trials per outcome category (made vs missed)** is ambitious but would start to give you a sense of whether ERPs are even plausible.

- **Spectral / time–frequency analyses:**
  - More tolerant to some noise, especially if you focus on **lower frequencies (theta/alpha)** and **average power within well-defined windows** (e.g., pre-shot baseline, recording window).
  - Because your per-trial segments are relatively long (several seconds), even **30–60 clean trials per condition** can already give **meaningful pilot-level insights**, especially for within-subject patterns.

**S:** So for this pilot, spectral is the primary target and ERPs are more exploratory.  
**E:** Exactly. That means we can **aim trial counts at spectral needs**, while keeping ERPs in mind as a “nice to have.”

---

### 6. Single-Participant Reliability Considerations

**S:** How much can I really conclude from just one person, no matter how many shots they take?  
**E:** That’s crucial:
- With **one participant**, you’re mostly assessing:
  - **Feasibility** (does the setup work in a real shooting context?).
  - **Data quality** (how noisy is Muse in this paradigm?).
  - **Rough effect patterns** (do we see any plausible alpha/theta changes from prep to shot?).
- You’re **not** establishing generalizable effects; you’re building a **proof-of-concept**.

**S:** So “reliability” here means more “within-subject stability” than statistical generalizability?  
**E:** Right. Within one participant:
- More trials improve the **stability of averaged power estimates**.
- You can also look at **test–retest consistency** if you split the data into sessions or halves and compare.

**S:** So I should bias toward having **enough trials to estimate within-subject patterns**, even if it’s only N=1.  
**E:** Exactly. That’s where **number of trials and number of sessions** come in.

---

### 7. Practical Fatigue Constraints for a Basketball Shooter

**S:** How many free throws can I reasonably ask them to take in one session?  
**E:** Think practically:
- A trained shooter can sometimes do **100–200 free throws in a practice** without major breakdown, but:
  - You need **consistent form** for your task, not just raw volume.
  - **Fatigue** will change both motor behavior and EEG (e.g., increased muscle tension, mental fatigue), which may or may not be desirable.

**S:** For this pilot, I’d like them reasonably fresh so the data reflects something like typical free throws, not end-of-practice exhaustion.  
**E:** Then a **single continuous block of 150–200 shots** might be too much:
- Attention drifts.
- Form deteriorates.
- The headband may need **periodic repositioning**.

**S:** Maybe something like **blocks of 40–60 shots** with breaks?  
**E:** That’s much more reasonable:
- For example, **3 blocks of ~40 shots** (about 120 total) with short rest and headband adjustment in between.
- Or **2 blocks of ~60 shots** if the shooter is strong and you want fewer context switches.

---

### 8. Working Toward a Trial Count Recommendation

**S:** Given all that, can we back-calculate a reasonable shot count?  
**E:** Let’s walk through it step-by-step for a **single session**:

1. **Target usable trials per condition (made vs missed)**  
   - Suppose you want **~40–50 clean trials per condition** for spectral analyses.
   - The challenge: free throws are not balanced; shooters might make, say, **60–80%** of shots.

2. **Account for shot outcomes**  
   - If the shooter makes **70%** of shots:
     - Out of 100 attempts: ~70 made, ~30 missed.
   - If you want **40 clean misses**, 100 total attempts is probably **not enough**.

3. **Account for artifact rejection (~50% usable)**  
   - Let’s say **about half of trials are clean enough** after quality control.
   - For 120 total shots:
     - You might get **~60 usable**.
     - If the shooter makes 70%, that’s roughly **42 made vs 18 missed** clean trials.

4. **Interpretation for the pilot**  
   - **42 clean made** and **18 clean missed** is:
     - Reasonable for **overall spectral analyses** (e.g., “all shots pooled” or “just made shots”).
     - Thin but not useless for **comparing made vs missed**, especially for exploratory purposes.

**S:** So with around 120 total attempts, I might end up with a strong signal for “all trials” and “made-only,” but only exploratory comparison for misses.  
**E:** Exactly. For a **single-participant pilot**, that’s a **realistic and useful outcome**.

**S:** And if I collected closer to 150–180 shots?  
**E:** Then, assuming similar make-rate and artifact profile:
- 150 total → ~75 usable → maybe **~50 made, ~25 missed**.
- 180 total → ~90 usable → maybe **~60 made, ~30 missed**.

That would give you **more comfortable numbers for spectral comparisons** between made and missed, and a better chance that even **simple ERP-style averages** look interpretable.

---

### 9. Considering Multiple Sessions

**S:** What about splitting this across multiple days?  
**E:** That can help with:
- **Fatigue management** – fewer shots per day, better quality.
- Checking **test–retest reliability** – do patterns replicate on a second day?

**S:** Any downsides?  
**E:** Mostly practical:
- More setup time, scheduling constraints.
- You need to ensure **similar conditions** (time of day, warm-up, environment) so sessions are comparable.

**S:** If I wanted ~150–180 total shots, how might I schedule that?  
**E:** A few examples:
- **Two sessions** of ~80–90 shots each. Each session could be **2–3 blocks of 30–40 shots**, with rest and headband checks between blocks.
- Or **three shorter sessions** of ~50–60 shots each if you’re especially concerned about fatigue or scheduling.

**S:** Would combining sessions still be meaningful with just one participant?  
**E:** Yes. You’d treat it as a **single-subject, multi-session dataset**, gaining:
- More total trials.
- A sense of **within-participant stability across days**.

---

### 10. Final Recommendation

**S:** Can you summarize what you’d recommend for my specific FreethrowEEG + Muse pilot?  
**E:** Sure. Based on:
- Typical EEG trial counts,
- Extra noise and movement artifacts with Muse during shooting,
- The dual interest in spectral measures and maybe basic ERPs,
- Single-participant reliability limits, and
- Practical fatigue constraints for a shooter,

I’d recommend the following:

- **Minimum number of shots (for a very basic pilot):**  
  - **~80 total shots** in one or two blocks.  
  - Expect maybe **35–45 usable trials** overall, which is enough to **test feasibility** and very rough spectral patterns, but limited for made vs missed comparisons.

- **Ideal number of shots (for a more informative pilot):**  
  - **~140–180 total shots** overall.  
  - This might look like **3 blocks of ~40–60 shots**, or spread across multiple sessions.  
  - With ~50% usable, you could get **~70–90 clean trials**, giving **solid spectral estimates** and at least **exploratory contrasts** between made and missed.

- **Number of sessions:**  
  - **1 session is acceptable** for a quick feasibility check (aim for ~100–120 shots).  
  - **2 sessions are preferred** if you target the ideal range (~150–180 total shots) while managing fatigue (e.g., **~80–90 shots per session**).  
  - **3 shorter sessions** can be used if endurance or scheduling is a concern, as long as conditions are reasonably matched.

**S:** That gives me a concrete plan: I’ll aim for about **150–180 total free throws**, spread over **2 sessions**, each with several shorter blocks and headband checks.  
**E:** Perfect. That should give you enough data to meaningfully test FreethrowEEG + Muse in a realistic shooting scenario, while keeping things feasible for both the participant and your analysis pipeline.

