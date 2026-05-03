"""
FreethrowEEG Video Analysis Pipeline Coordinator
Orchestrates video synchronization, pose estimation, and figure generation.
Bridges data formats between the component scripts and runs the full pipeline.

Supports multi-block sessions: loads all blocks, runs video analysis on blocks
that have associated video files (currently Block 3), and generates figures
using combined pose + EEG data.
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
    detect_release_from_full_window,
    extract_release_aligned_epoch,
    VIDEO_FPS,
)
from video_figures import generate_all_figures
from load_multiblock import (
    load_session_multiblock, VIDEO_FILES, BLOCK_FILES,
)


def _build_video_figures_pose_data(all_features, pose_results, fps):
    """Convert pose_analysis output into the format video_figures expects.

    Handles both release-aligned features (with timestamps relative to release)
    and legacy recording-window features.
    """
    converted = {}
    for sn, feats in all_features.items():
        if feats is None:
            converted[sn] = {}
            continue

        frames = feats['frames']
        if 'timestamps' in feats and hasattr(feats['timestamps'], '__len__'):
            ts = np.asarray(feats['timestamps'])
        elif len(frames) > 0:
            ts = (frames - frames[0]) / fps
        else:
            ts = np.array([])

        entry = {
            'timestamps': ts.tolist() if hasattr(ts, 'tolist') else list(ts),
            'elbow_angle': feats['elbow_angle'].tolist() if hasattr(feats['elbow_angle'], 'tolist') else list(feats['elbow_angle']),
            'wrist_height': feats['wrist_height'].tolist() if hasattr(feats['wrist_height'], 'tolist') else list(feats['wrist_height']),
            'knee_angle': feats['knee_angle'].tolist() if hasattr(feats['knee_angle'], 'tolist') else list(feats['knee_angle']),
            'body_lean': feats.get('body_lean_angle', feats.get('body_lean', [])),
            'shoulder_angle': feats.get('shoulder_angle', []),
            'center_of_mass_y': feats.get('center_of_mass_y', []),
            'release_frame': feats.get('release_frame'),
            'release_idx_in_epoch': feats.get('release_idx_in_epoch'),
            'wrist_range': feats.get('wrist_range'),
        }
        for k in ('body_lean', 'shoulder_angle', 'center_of_mass_y'):
            v = entry[k]
            if hasattr(v, 'tolist'):
                entry[k] = v.tolist()
            elif not isinstance(v, list):
                entry[k] = list(v) if v is not None else []

        # Attach raw landmarks from both recording and full_window
        rec_lm = pose_results.get(sn, {}).get('recording', {})
        full_lm = pose_results.get(sn, {}).get('full_window', {})
        lm_source = rec_lm if rec_lm else full_lm
        entry['raw_landmarks'] = {str(k): v.tolist() for k, v in lm_source.items()}

        converted[sn] = entry

    return converted


def _build_block_session(session, block_num, block_info):
    """Create a session-like dict containing only shots from a specific block,
    with timestamps converted to block-local time."""
    bi = block_info[block_num]
    time_offset = bi['time_offset']

    block_shots = [s for s in session['shots'] if s.get('block') == block_num]

    local_shots = []
    for shot in block_shots:
        local_shot = dict(shot)
        local_shot['duration'] = shot['duration'] - time_offset

        local_eeg = {}
        for phase in shot['eegData']:
            local_eeg[phase] = {}
            for band in shot['eegData'][phase]:
                local_eeg[phase][band] = []
                for entry in shot['eegData'][phase][band]:
                    local_eeg[phase][band].append({
                        'timestamp': entry['timestamp'] - time_offset,
                        'power': entry['power'],
                    })
        local_shot['eegData'] = local_eeg
        local_shots.append(local_shot)

    block_session = {
        'playerName': session.get('playerName', 'Unknown'),
        'totalShots': len(local_shots),
        'sessionDuration': bi['sample_duration_sec'],
        'shots': local_shots,
    }
    return block_session


def run_pipeline(data_path=None, video_path=None, output_dir=None,
                 skip_clips=False):
    """Run the complete video analysis pipeline.

    When called without arguments, loads the multi-block session and processes
    all blocks that have video files (Block 3 by default).
    """
    if output_dir is None:
        output_dir = SCRIPT_DIR / 'video_output'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = SCRIPT_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading session data")
    print("=" * 60)
    session, block_info = load_session_multiblock()

    video_blocks = []
    if video_path is not None:
        video_blocks = [(3, Path(video_path))]
    else:
        for bnum in sorted(block_info.keys()):
            bi = block_info[bnum]
            if bi.get('has_video') and bi['valid_shots'] > 3:
                video_blocks.append((bnum, Path(bi['video_path'])))

    if not video_blocks:
        print("  WARNING: No blocks with video found. Skipping video pipeline.")
        return None

    print(f"  Blocks with video: {[b[0] for b in video_blocks]}")

    combined_pose_results = {}
    combined_features = {}
    combined_shot_times = []
    primary_video_path = None

    for block_num, vid_path in video_blocks:
        bi = block_info[block_num]
        time_offset = bi['time_offset']

        print(f"\n{'=' * 60}")
        print(f"Processing Block {block_num} (video: {vid_path.name})")
        print(f"{'=' * 60}")

        block_session = _build_block_session(session, block_num, block_info)
        block_shot_times = get_shot_times(block_session, time_offset=0.0)

        n_made = sum(1 for s in block_shot_times if s['success'])
        n_missed = sum(1 for s in block_shot_times if not s['success'])
        print(f"  Shots: {len(block_shot_times)} ({n_made} made, {n_missed} missed)")

        for st in block_shot_times:
            label = 'made' if st['success'] else 'missed'
            print(f"  Shot {st['shot_number']:2d} ({label:6s}): "
                  f"rec={st['recording_start']:.1f}–{st['recording_end']:.1f}s")

        if not skip_clips:
            print(f"\n  Extracting clips and frames...")
            run_sync(None, vid_path, output_dir, session=block_session,
                     shot_times=block_shot_times)
        else:
            print("  Skipping clip extraction (--skip-clips)")

        print(f"\n  Running MediaPipe pose estimation...")
        pose_results = run_pose_estimation(vid_path, block_shot_times)

        cap_for_fps = __import__('cv2').VideoCapture(str(vid_path))
        fps = cap_for_fps.get(__import__('cv2').CAP_PROP_FPS) or VIDEO_FPS
        cap_for_fps.release()

        print(f"\n  Extracting biomechanical features (full window)...")
        block_features_full = {}
        block_features = {}
        n_aligned = 0
        n_fallback = 0
        for st in block_shot_times:
            sn = st['shot_number']
            shot_data = pose_results.get(sn, {})

            # Use full_window (prep→review) for feature extraction
            lm_full = shot_data.get('full_window', {})
            lm_rec = shot_data.get('recording', {})

            feats_full = extract_pose_features(lm_full) if lm_full else None
            block_features_full[sn] = feats_full

            release_info = detect_release_from_full_window(feats_full, fps)

            if release_info is not None:
                aligned = extract_release_aligned_epoch(
                    feats_full, release_info, fps,
                    pre_sec=2.0, post_sec=2.0
                )
                aligned['release_frame'] = release_info['release_frame']
                block_features[sn] = aligned
                n_aligned += 1
                n_full = len(feats_full['frames']) if feats_full else 0
                n_epoch = len(aligned['frames'])
                print(f"    Shot {sn}: full={n_full}fr, release-aligned={n_epoch}fr "
                      f"(wrist_range={release_info['wrist_range']:.3f})")
            else:
                # Fallback to recording-window features
                feats_rec = extract_pose_features(lm_rec) if lm_rec else None
                block_features[sn] = feats_rec
                n_fallback += 1
                n_frames = len(feats_rec['frames']) if feats_rec else 0
                wr = (max(feats_rec['wrist_height']) - min(feats_rec['wrist_height'])
                      ) if feats_rec and len(feats_rec['wrist_height']) > 0 else 0
                print(f"    Shot {sn}: no clear release detected, "
                      f"fallback to recording window ({n_frames}fr, "
                      f"wrist_range={wr:.3f})")

        print(f"\n  Release-aligned: {n_aligned}, fallback: {n_fallback}")

        combined_pose_results.update(pose_results)
        combined_features.update(block_features)
        combined_shot_times.extend(block_shot_times)

        if primary_video_path is None:
            primary_video_path = vid_path

    print(f"\n{'=' * 60}")
    print("Comparing made vs missed across all video blocks...")
    print(f"{'=' * 60}")
    comparison = compare_made_vs_missed(combined_features, combined_shot_times)

    for feat_key, comp in comparison.items():
        made_n = comp.get('made_n', 0)
        missed_n = comp.get('missed_n', 0)
        if feat_key != 'release_frame' and made_n > 0 and missed_n > 0:
            made_peak = comp.get('made_peak_mean', float('nan'))
            missed_peak = comp.get('missed_peak_mean', float('nan'))
            print(f"  {feat_key}: made peak={made_peak:.2f}, missed peak={missed_peak:.2f}")

    pose_dir = output_dir / 'pose'
    pose_dir.mkdir(parents=True, exist_ok=True)
    print("\n  Saving annotated key frames...")
    saved_frames = _save_annotated_keyframes(
        primary_video_path, combined_pose_results, combined_features,
        combined_shot_times, pose_dir
    )
    print(f"  Saved {len(saved_frames)} key-frame images")

    print(f"\n{'=' * 60}")
    print("Generating publication figures (9-13)")
    print(f"{'=' * 60}")

    video_session = _build_block_session(
        session, video_blocks[0][0], block_info
    )

    pose_data_for_figs = _build_video_figures_pose_data(
        combined_features, combined_pose_results, fps
    )

    figures = generate_all_figures(
        video_session, combined_shot_times, pose_data_for_figs,
        primary_video_path, fig_dir
    )

    print(f"\n{'=' * 60}")
    print("Saving combined results")
    print(f"{'=' * 60}")

    n_made_total = sum(1 for s in combined_shot_times if s['success'])
    n_missed_total = sum(1 for s in combined_shot_times if not s['success'])

    results = {
        'session_file': 'multiblock',
        'video_files': {str(bnum): str(vp) for bnum, vp in video_blocks},
        'n_shots': len(combined_shot_times),
        'n_made': n_made_total,
        'n_missed': n_missed_total,
        'shot_times': combined_shot_times,
        'features_per_shot': _features_to_serialisable(combined_features),
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
            'features_per_shot': _features_to_serialisable(combined_features),
            'comparison': _comparison_to_serialisable(comparison),
        }, f, indent=2, default=str)
    print(f"  Pose features saved to {pose_json_path}")

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Output directory: {output_dir}")
    print(f"  Figures directory: {fig_dir}")
    print(f"  Generated figures:")
    for name, fpath in figures.items():
        print(f"    {name}: {Path(fpath).name}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run complete FreethrowEEG video analysis pipeline')
    parser.add_argument('data_file', nargs='?', default=None,
                        help='Path to session JSON (default: load multi-block)')
    parser.add_argument('video_file', nargs='?', default=None,
                        help='Path to video file (default: auto-detect)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: analysis/video_output/)')
    parser.add_argument('--skip-clips', action='store_true',
                        help='Skip clip/frame extraction')
    args = parser.parse_args()

    run_pipeline(args.data_file, args.video_file, args.output,
                 skip_clips=args.skip_clips)
