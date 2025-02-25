# Graduate Student Meeting Simulation: Planning the FreethrowEEG Study

## Meeting Transcript

_Setting: Lab meeting room, 2:30 PM. Sarah (basketball/sports science expert) and Alex (EEG/programming expert) sit down with their laptops and coffee._

Sarah: "Thanks for meeting to discuss this freethrow study. I've been thinking a lot about the experimental design."
_(Internally: Hoping the technical constraints won't limit what we can do with the basketball elements)_

Alex: "Of course! I've been looking at the EEG setup, and I think we can get some really clean data."
_(Internally: Please don't suggest anything too complicated that will make the signal processing impossible)_

Sarah: "So, I've been thinking about shot volume. In basketball training, we typically want at least 100 shots per session to see meaningful patterns in form and accuracy."
_(Internally: Actually, we'd prefer 300+ shots, but that might be too much with all this equipment)_

Alex: _(Wincing slightly)_ "A hundred shots? How long does each shot typically take? We need to consider fatigue effects on the EEG data."
_(Internally: That's going to be a massive amount of data to process and clean)_

Sarah: "Well, in a normal practice, players take about 8-10 seconds between shots. But with the EEG, we probably need more time to get a clean baseline?"
_(Internally: Please don't say we need like a minute between shots)_

Alex: "Actually, that might work perfectly. I'd suggest 15 seconds total per trial: 5 seconds baseline, then the shot itself, then a few seconds after. That way we can capture the pre-shot mental state and any post-shot processing."
_(Internally: Relief that the timing actually works well with EEG epoch requirements)_

Sarah: "That's not bad at all. So for 100 shots, we're looking at about 25-30 minutes of actual shooting time. What about electrode impedance checks and setup?"
_(Internally: Hoping this won't turn into a two-hour session)_

Alex: "Initial setup usually takes about 15 minutes once someone's practiced. We should probably do impedance checks every 25 shots or so, maybe during brief water breaks?"
_(Internally: Worried about sweat affecting the signal quality during long sessions)_

Sarah: "That fits well with basketball training anyway - we typically break every 25-30 shots to prevent fatigue. How many sessions do you think we need for good data?"
_(Internally: Hoping for enough sessions to show learning effects)_

Alex: "For meaningful EEG patterns, especially if we're looking at learning effects, I'd say minimum 10 sessions. More would be better."
_(Internally: Already dreading the hours of preprocessing)_

Sarah: "What if we did 20 sessions over 4 weeks? Five sessions per week, Monday through Friday, same time each day to control for circadian effects?"
_(Internally: Excited about the potential for a solid longitudinal dataset)_

Alex: _(Brightening)_ "That would be perfect actually. We could treat each week as a block and look for both within-session and across-week changes in brain patterns."
_(Internally: This could make a really nice paper)_

Sarah: "For each shot, what does the participant need to do besides, well, shooting? Any specific markers or triggers we need?"
_(Internally: Hoping the technical requirements won't interfere with natural shooting motion)_

Alex: "They'll need to stand in a consistent spot for EEG purposes. I'm thinking we have them:

1. Get in position
2. Stand still for 5 seconds (baseline)
3. Take the shot when they hear a soft beep
4. Return to position
   Does that work from a basketball perspective?"
   _(Internally: Please say this won't mess up their shooting rhythm)_

Sarah: "That could work, especially if we give them some practice trials to get used to the rhythm. Should we record makes and misses manually or can we automate that?"
_(Internally: Remembering previous manual scoring nightmares)_

Alex: "We could set up a simple key press system - experimenter hits one key for make, another for miss. Much more reliable than trying to automate it with video processing."
_(Internally: Relieved to avoid complex video analysis)_

Sarah: "Perfect. What about individual differences? Should we try to get multiple participants?"
_(Internally: Hoping to increase the sample size)_

Alex: "For a first study, I'd actually suggest going deep with one good participant. Twenty sessions of clean data from one person is more valuable than noisy data from multiple people. We can always expand later."
_(Internally: One participant means consistent electrode placement and less variance to deal with)_

Sarah: "That makes sense. We could look for someone who:

- Plays basketball regularly but isn't varsity level
- Has consistent form
- Can commit to 4 weeks
- Ideally has short hair for EEG purposes"
  _(Internally: Already thinking of several potential participants)_

Alex: "Exactly. And we should probably do a pre-screening session to check their baseline shooting percentage and make sure they're comfortable with the EEG setup."
_(Internally: Hoping to avoid discovering setup issues during the actual study)_

## Summary Document for Advisor

### FreethrowEEG Study Design

#### Participant Characteristics

- Single participant pilot study
- Regular basketball player (non-varsity)
- Consistent shooting form
- Available for 4 consecutive weeks
- Pre-screened for EEG compatibility and baseline performance

#### Session Structure

- 20 sessions total (5 sessions per week, Monday-Friday)
- Consistent time of day across sessions
- 100 freethrows per session
- Each session approximately 45 minutes:
  - 15 minutes setup/preparation
  - 25-30 minutes shooting
  - Brief breaks every 25 shots

#### Trial Structure

1. Participant takes position at freethrow line
2. 5-second baseline period (standing still)
3. Auditory cue for shot
4. Shot execution
5. Brief return to position
6. Manual recording of shot outcome
   Total trial time: ~15 seconds

#### Data Collection

- Continuous EEG recording
- Manual logging of shot success/failure
- Session metadata (time, date, environmental conditions)
- Brief pre/post session questionnaires
- Regular impedance checks during breaks

#### Expected Outcomes

1. Within-session analysis:

   - Changes in brain activity patterns
   - Fatigue effects
   - Shot success rate variations

2. Across-session analysis:

   - Learning effects
   - Stability of brain patterns
   - Performance improvements

3. Potential Measures:
   - Alpha power during baseline
   - Pre-shot neural patterns
   - Correlation between brain states and success
   - Learning curve analysis

#### Timeline

- Week 0: Participant screening and baseline
- Weeks 1-4: Data collection
- Weeks 5-6: Initial analysis and review
- Week 7: Decision point for study expansion

#### Resource Requirements

- Muse EEG system
- Basketball court (consistent location)
- Data collection computer
- Research assistant for manual scoring
- Participant compensation

_Note: This design prioritizes data quality and longitudinal consistency over sample size, allowing us to establish solid methodological foundations for future larger-scale studies._
