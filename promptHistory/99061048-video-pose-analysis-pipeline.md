# Chat: Video Pose Analysis Pipeline

*Transcript ID: 99061048-a39c-40d6-8de0-c8c9b266168f*

---

## User

Thanks for analysing the data in @data/freethrow_session_lukas_test_2026-02-16.json and creating hte report in @analysis with a set of analysis scripts. i am found a video that was recorded during the session and I want to consider using modern AI pose estimation tools to identify the pose features of each shot and match them with the concomitantly recorded EEG data. We also know if each shot was success or failure,

I am not sure how we have synced the video and the shots, but perahps we can hear the beeps from the trails in the video audio stream @data/freethrow_session_lukas_test_2026-02-16.webm

Add you analysis to the analysis steps we already used to make the paper, and supplment and commpliment existing analysis and stats with this video evidence, At the least pull a few still shots in the shot process (like a stop motion series of shots of a shot). Consider cutting clips of each success and fail shot and then doing signal processing on that video (average shot, ghost shots, etc).

Use multiagents to delegate jobs and combine their work with a coordinator bot

---

## Assistant

I'll start by understanding the existing analysis pipeline and data structure, then coordinate multiple agents to tackle the video analysis tasks in parallel. Let me first explore what we're working with.

The user wants me to:

1. Analyze a video recording from a freethrow session
2. Use AI pose estimation to identify pose features for each shot
3. Sync the video with the shot data (possibly using audio beeps)
4. Match pose features with EEG data
5. Create still shots (stop motion series) of shots
6. Cut clips of success/fail shots
7. Do signal processing on video (average shot, ghost shots)
8. Add this to existing analysis pipeline
9. Use multiagents to delegate and coordinate

Let me first understand the existing data and analysis structure before launching agents.

---

## Assistant

Good, I have a clear picture of the project. Now let me understand how the video was recorded and synced, then coordinate the multi-agent work.

Now I have a good understanding of the project. Let me plan the multi-agent approach:

1. **Audio Sync Agent**: Extract audio from the webm, find beeps to sync trials with video timestamps
2. **Video Processing Agent**: Extract frames, cut clips per shot, create stop-motion sequences
3. **Pose Estimation Agent**: Use MediaPipe or similar for pose estimation on extracted frames
4. **Analysis Integration Agent**: Combine video/pose data with EEG data, create new figures

Let me first understand the web app code to see how the video recording was started/stopped relative to the trials, which will help with synchronization.

I should also check the HTML file to see what sync details are embedded there.

---

## Assistant

I can see from the HTML source that the camera was captured with `audio: false`, so the beeps won't be in the video. However, the session JSON has precise timestamps for each shot phase, and the video recording started/stopped with the session, so we can sync by matching timelines directly. Let me check the timing configuration and video properties.

Now I have a thorough understanding. Key observations:

1. **Video sync challenge**: The camera is captured with `audio: false` - so NO audio in the video. The beeps are played through the system speaker via AudioContext. So we can't use audio from the video to sync.

2. **But** we have timing data in the JSON - each shot has timestamps for each phase. And the video is a continuous recording of the session. So we need to figure out the video start time relative to the session start time.

3. **Beeps per phase**:
   - prep: 800Hz, 0.2s
   - preShot: 900Hz, 0.2s
   - recording: 1200Hz, 0.2s
   - postShot: 700Hz, 0.2s
   - review: 600Hz, 0.3s

4. **Phase timing from CONFIG** - I need to check the CONFIG object for the timeout durations. Since audio is `false` in the camera capture, the webm likely has no audio track. But wait - there might still be system audio captured somehow, or maybe the webm was recorded differently. Let me check the actual code to see how the video recording is set up.

Looking at both the freethrow_eeg_working.html and index.html files, they both have `audio: false` in the camera configuration, so there's definitely no audio in the video. However, the session JSON has precise timestamps for each shot and phase, which means I can sync everything if I can figure out when the video recording started relative to the session start. The session data includes the session start timestamp, completion time, and total duration, plus individual timestamps for each shot, so the video recording likely started and stopped with the session itself. Now I need to check how the video recording is actually initiated and stopped in the code.

Looking at the HTML, it appears `startRecording()` gets called when the session begins and `stopRecording()` when it ends, so the video timeline should align with the session timeline. I should look more carefully at where these functions are called and also check the CONFIG for phase durations.

My approach will be to set up a virtual environment with the necessary packages like mediapipe and opencv, then probe the video to determine its duration and attempt syncing with the session JSON timestamps. Even without audio, I can match the video duration against the session duration. Then I'll extract individual frames for each shot to create the stop-motion sequence, cut clips per shot, and run MediaPipe pose estimation on each shot.

Now I'm thinking through the remaining steps: creating composite visualizations like average shots and ghost overlays, then integrating everything with the existing EEG analysis. Before diving into the full pipeline, I should check the video properties and duration, understand the timing structure from the CONFIG and HTML, set up the virtual environment, and extract the shot timestamps from the JSON so I can coordinate the parallel agents for the main analysis work.

---

## Assistant

Now I have a complete understanding of the project. Key insight: the camera was captured with `audio: false`, but the JSON has precise timestamps for each shot phase, and the video recording started when the session started - so shot `duration` fields map directly to video timestamps.

Let me set up the environment and then coordinate multiple agents for the video analysis pipeline.

Now I have a comprehensive understanding of the project. Let me plan the multi-agent workflow:

Key findings:
1. Video was captured with `audio: false` - so no audio beeps in video
2. The session JSON has timestamps for each shot and each phase
3. Shot phase timing: prep=5s, preShot=10s, recording/shot=3s, postShot=5s (total ~23s per shot cycle)
4. Session started at "2026-02-16T01:09:16.419Z", completed at "2026-02-16T01:15:04.428Z" (348s)

5. Shot timestamps in the JSON are relative to session start, and the video recording began when the session started
6. There are 10 total shots with 3 makes and 7 misses

For the multi-agent coordination, I'll have Agent 1 set up the environment and probe the video file to verify sync between video duration and session timing. Agent 2 will handle the video processing—extracting frames per shot, creating stop-motion sequences, and cutting individual shot clips. Agent 3 will write the pose estimation script to analyze player movement.

Agent 4 will create composite visualizations combining pose data with EEG information and generate updated figures for the paper.

*[Additional planning details about video sync, shot timing, pipeline architecture, and dependencies...]*

---

## Assistant

Python 3.9 is too old for current mediapipe. Let me find a newer Python version.

The Python 3.9 in the venv is old. Let me check if there's a newer Python available on this system.

---

## Assistant

The video is 347.4s and the session is 348.01s - almost perfect alignment. The video timestamps map directly to the session timestamps. Now let me launch parallel agents to write the three video analysis scripts.

*[Shot timing details and agent coordination...]*

---

## Assistant

All three scripts are written. Let me verify them and run the pipeline. First, let me check the scripts for any issues.

All three scripts have been written. Now let me verify them, then try running them.

---

## Assistant

The scripts look solid but I see some data format compatibility issues between them. Let me fix the interfaces and write a coordinator script to tie everything together.

*[Format mismatches between pose_analysis.py, video_sync.py, video_figures.py - coordinator script approach...]*

---

## Assistant

Now let me update the feature extraction to also return timestamps in seconds (not just frame indices):

---

## Assistant

Now let me fix `video_figures.py` so the ghost overlay works with the actual pose landmark data structure from `pose_analysis.py`. The landmarks stored by pose_analysis are `{frame_idx: (33,4) array}`, not a simple list:

---

## Assistant

Now let me write the coordinator script that ties everything together:

---

## Assistant

Now let me run the pipeline. This will be a long-running process due to pose estimation processing every frame:

---

## Assistant

MediaPipe's API has changed in newer versions. Let me check and adapt:

The newer version of mediapipe (0.10.32) has changed its API. Let me check what version we have and how to use it.

---

## Assistant

Good, now I need to rewrite `pose_analysis.py` to use the new MediaPipe Tasks API:

---

## Assistant

Now update the `video_figures.py` connections list to use the new API:

---

## Assistant

Now let's run the pipeline:

---

## Assistant

The entire pipeline ran successfully. Let me verify the outputs and then integrate into the report.

---

## Assistant

Excellent! Everything generated successfully: 13 figures total, 10 shot clips, 20 montages, 19 annotated keyframes, and 140 still frames. Now let me integrate this into the report pipeline.

*[Report template updates, requirements.txt, generate_report.py updates...]*

---

## Assistant

Everything is complete. Here's a summary of what was built and generated:

---

## Video Analysis Pipeline - Complete

### Architecture (Multi-Agent Coordinated)

Four new scripts were created in `analysis/`, orchestrated by a coordinator:

| Script | Purpose |
|---|---|
| `video_sync.py` | Syncs video with session JSON, extracts per-shot clips and still frames |
| `pose_analysis.py` | Runs MediaPipe Pose (heavy model) on all 10,422 frames, extracts biomechanical features |
| `video_figures.py` | Generates 5 publication-quality figures (9-13) combining video, pose, and EEG |
| `run_video_analysis.py` | Coordinator that orchestrates the full pipeline and bridges data formats |

### Key Finding: Video-EEG Sync

The video (347.4s) and session JSON (348.0s) are nearly perfectly aligned - timestamps in the JSON map directly to video time since both recording streams started together. No audio-based sync was needed.

### Generated Outputs

**13 Publication Figures** (up from 8):
- **Figs 1-8**: Original EEG analysis (unchanged)
- **Fig 9** (`fig_shot_montage.pdf`): Stop-motion strip of all 10 shots with pose skeleton overlays
- **Fig 10** (`fig_ghost_overlay.pdf`): Ghost composites at release point — made vs missed
- **Fig 11** (`fig_pose_kinematics.pdf`): Elbow angle, wrist height, knee angle, body lean trajectories
- **Fig 12** (`fig_pose_eeg_combined.pdf`): Time-aligned pose kinematics + EEG (alpha, theta/alpha)
- **Fig 13** (`fig_average_frames.pdf`): Pixel-averaged frames at start/mid/end with difference maps

**Video Clips**: 10 individual shot clips + `all_made.mp4` + `all_missed.mp4`
**Still Frames**: 140 frames (8 recording + 6 key moments per shot)
**Montages**: 20 film-strip montages (recording phase + key moments per shot)
**Annotated Keyframes**: 19 frames with pose skeleton overlays
**16-page PDF Report**: Updated with video/pose analysis sections

### Pose Estimation Results

MediaPipe detected poses in 31% of frames (3,264/10,427). Key biomechanical differences between made and missed shots:

| Feature | Made Peak | Missed Peak |
|---|---|---|
| Elbow angle | 167.4° | 178.5° |
| Shoulder angle | 35.3° | 79.7° |
| Wrist height | 0.47 | 0.22 |
| Body lean | 88.8° | 43.6° |

### Running the Pipeline

```bash
source .venv/bin/activate
python analysis/run_video_analysis.py \
  data/freethrow_session_lukas_test_2026-02-16.json \
  data/freethrow_session_lukas_test_2026-02-16.webm

python analysis/generate_report.py \
  data/freethrow_session_lukas_test_2026-02-16.json
```
