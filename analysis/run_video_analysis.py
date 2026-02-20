"""
FreethrowEEG Video Analysis Pipeline Coordinator
Orchestrates video synchronization, pose estimation, and figure generation.
Bridges data formats between the component scripts and runs the full pipeline.
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from video_sync import load_session, get_shot_times, run_sync
from pose_analysis import (
    run_pose_estimation,
    extract_pose_features,
    compare_made_vs_missed,
    draw_pose_on_frame,
    _save_annotated_keyframes,
    _features_to_serialisable,
    _comparison_to_serialisable,
    VIDEO_FPS,
)
from video_figures import generate_all_figures


def _build_video_figures_pose_data(all_features, pose_results, fps):
    """Convert pose_analysis output into the format video_figures expects.

    video_figures.py expects per-shot dicts with:
      - 'timestamps': array of seconds (relative to recording start)
      - 'elbow_angle', 'wrist_height', 'knee_angle', 'body_lean': arrays
      - 'raw_landmarks': {frame_idx: (33,4) array} for skeleton drawing
    """
    converted = {}
    for sn, feats in all_features.items():
        if feats is None:
            converted[sn] = {}
            continue

        frames = feats['frames']
        if len(frames) > 0:
            ts = (frames - frames[0]) / fps
        else:
            ts = np.array([])

        entry = {
            'timestamps': ts.tolist(),
            'elbow_angle': feats['elbow_angle'].tolist(),
            'wrist_height': feats['wrist_height'].tolist(),
            'knee_angle': feats['knee_angle'].tolist(),
            'body_lean': feats['body_lean_angle'].tolist(),
            'shoulder_angle': feats['shoulder_angle'].tolist(),
            'center_of_mass_y': feats['center_of_mass_y'].tolist(),
            'release_frame': feats.get('release_frame'),
        }

        rec_lm = pose_results.get(sn, {}).get('recording', {})
        entry['raw_landmarks'] = {str(k): v.tolist() for k, v in rec_lm.items()}

        converted[sn] = entry

    return converted


def run_pipeline(data_path, video_path, output_dir=None, skip_clips=False):
    """Run the complete video analysis pipeline.

    Steps:
      1. Load session data and extract shot timing
      2. Extract video clips and still frames
      3. Run MediaPipe pose estimation
      4. Extract biomechanical features
      5. Generate publication figures (9-13)
      6. Save combined results JSON
    """
    data_path = Path(data_path)
    video_path = Path(video_path)

    if output_dir is None:
        output_dir = SCRIPT_DIR / 'video_output'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = SCRIPT_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load and sync ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading session data and extracting shot timing")
    print("=" * 60)
    session = load_session(data_path)
    shot_times = get_shot_times(session)
    n_made = sum(1 for s in shot_times if s['success'])
    n_missed = sum(1 for s in shot_times if not s['success'])
    print(f"  Player: {session['playerName']}")
    print(f"  Shots: {len(shot_times)} ({n_made} made, {n_missed} missed)")

    for st in shot_times:
        label = 'made' if st['success'] else 'missed'
        print(f"  Shot {st['shot_number']:2d} ({label:6s}): "
              f"rec={st['recording_start']:.1f}–{st['recording_end']:.1f}s")

    # ── Step 2: Extract clips and frames ──────────────────────────────────
    if not skip_clips:
        print("\n" + "=" * 60)
        print("STEP 2: Extracting video clips and still frames")
        print("=" * 60)
        run_sync(data_path, video_path, output_dir)
    else:
        print("\n  Skipping clip extraction (--skip-clips)")

    # ── Step 3: Pose estimation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Running MediaPipe pose estimation")
    print("=" * 60)
    pose_results = run_pose_estimation(video_path, shot_times)

    # ── Step 4: Feature extraction ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Extracting biomechanical features per shot")
    print("=" * 60)
    cap_for_fps = __import__('cv2').VideoCapture(str(video_path))
    fps = cap_for_fps.get(__import__('cv2').CAP_PROP_FPS) or VIDEO_FPS
    cap_for_fps.release()

    all_features = {}
    for st in shot_times:
        sn = st['shot_number']
        shot_data = pose_results.get(sn, {})
        lm_seq = shot_data.get('recording', {})
        if not lm_seq:
            lm_seq = shot_data.get('full_window', {})
        feats = extract_pose_features(lm_seq)
        all_features[sn] = feats
        n_frames = len(feats['frames']) if feats else 0
        print(f"  Shot {sn}: {'OK' if feats else 'no pose data'} ({n_frames} frames)")

    print("\n  Comparing made vs missed...")
    comparison = compare_made_vs_missed(all_features, shot_times)
    for feat_key, comp in comparison.items():
        made_n = comp.get('made_n', 0)
        missed_n = comp.get('missed_n', 0)
        if feat_key != 'release_frame' and made_n > 0 and missed_n > 0:
            made_peak = comp.get('made_peak_mean', float('nan'))
            missed_peak = comp.get('missed_peak_mean', float('nan'))
            print(f"    {feat_key}: made peak={made_peak:.2f}, missed peak={missed_peak:.2f}")

    # Save annotated keyframes
    pose_dir = output_dir / 'pose'
    pose_dir.mkdir(parents=True, exist_ok=True)
    print("\n  Saving annotated key frames...")
    saved_frames = _save_annotated_keyframes(
        video_path, pose_results, all_features, shot_times, pose_dir
    )
    print(f"  Saved {len(saved_frames)} key-frame images")

    # ── Step 5: Generate figures ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Generating publication figures (9-13)")
    print("=" * 60)

    pose_data_for_figs = _build_video_figures_pose_data(
        all_features, pose_results, fps
    )

    figures = generate_all_figures(
        session, shot_times, pose_data_for_figs, video_path, fig_dir
    )

    # ── Step 6: Save results ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Saving combined results")
    print("=" * 60)

    results = {
        'session_file': str(data_path),
        'video_file': str(video_path),
        'n_shots': len(shot_times),
        'n_made': n_made,
        'n_missed': n_missed,
        'shot_times': shot_times,
        'features_per_shot': _features_to_serialisable(all_features),
        'comparison': _comparison_to_serialisable(comparison),
        'figures': {k: str(v) for k, v in figures.items()},
        'keyframes': [str(p) for p in saved_frames],
    }

    results_path = output_dir / 'video_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")

    pose_json_path = pose_dir / 'pose_features.json'
    with open(pose_json_path, 'w') as f:
        json.dump({
            'features_per_shot': _features_to_serialisable(all_features),
            'comparison': _comparison_to_serialisable(comparison),
        }, f, indent=2, default=str)
    print(f"  Pose features saved to {pose_json_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Output directory: {output_dir}")
    print(f"  Figures directory: {fig_dir}")
    print(f"  Generated figures:")
    for name, fpath in figures.items():
        print(f"    {name}: {Path(fpath).name}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run complete FreethrowEEG video analysis pipeline')
    parser.add_argument('data_file', help='Path to session JSON file')
    parser.add_argument('video_file', help='Path to session video file')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: analysis/video_output/)')
    parser.add_argument('--skip-clips', action='store_true',
                        help='Skip clip/frame extraction (useful for re-running figures)')
    args = parser.parse_args()

    run_pipeline(args.data_file, args.video_file, args.output,
                 skip_clips=args.skip_clips)
