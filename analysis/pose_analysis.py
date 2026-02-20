"""
FreethrowEEG Pose Analysis
Runs MediaPipe pose estimation on basketball free-throw video clips and
extracts biomechanical features for comparing made vs missed shots.
"""

import json
import sys
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions
PoseLandmark = mp.tasks.vision.PoseLandmark
PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections

POSE_CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]

RIGHT_SHOULDER = PoseLandmark.RIGHT_SHOULDER  # 12
RIGHT_ELBOW = PoseLandmark.RIGHT_ELBOW        # 14
RIGHT_WRIST = PoseLandmark.RIGHT_WRIST        # 16
RIGHT_HIP = PoseLandmark.RIGHT_HIP            # 24
RIGHT_KNEE = PoseLandmark.RIGHT_KNEE          # 26
RIGHT_ANKLE = PoseLandmark.RIGHT_ANKLE        # 28
LEFT_HIP = PoseLandmark.LEFT_HIP              # 23
LEFT_SHOULDER = PoseLandmark.LEFT_SHOULDER     # 11

MODEL_PATH = Path(__file__).parent / 'pose_landmarker_heavy.task'
PHASE_ORDER = ['prep', 'preShot', 'recording', 'postShot', 'review']
VIDEO_FPS = 30


# ── Import / fallback for shot timing ────────────────────────────────────────

def _try_import_video_sync():
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from video_sync import get_shot_times as _gst, load_session as _ls
        return _gst, _ls
    except ImportError:
        return None, None


def load_session(filepath):
    _get_shot_times, _load_session = _try_import_video_sync()
    if _load_session is not None:
        return _load_session(filepath)
    with open(filepath) as f:
        return json.load(f)


def get_shot_times(session):
    _get_shot_times, _ = _try_import_video_sync()
    if _get_shot_times is not None:
        return _get_shot_times(session)

    shots_info = []
    for shot in session.get('shots', []):
        phases = {}
        for phase_name in PHASE_ORDER:
            phase_data = shot.get('eegData', {}).get(phase_name, {})
            all_timestamps = []
            for band_data in phase_data.values():
                if isinstance(band_data, list):
                    all_timestamps.extend(e['timestamp'] for e in band_data if 'timestamp' in e)
            if all_timestamps:
                phases[phase_name] = {
                    'start_time': min(all_timestamps),
                    'end_time': max(all_timestamps),
                }
        prep_ts = phases.get('prep', {}).get('start_time')
        rec_ts = phases.get('recording', {})
        last_phase = None
        for p in reversed(PHASE_ORDER):
            if p in phases:
                last_phase = phases[p]
                break

        shots_info.append({
            'shot_number': shot.get('shotNumber', len(shots_info) + 1),
            'success': shot.get('success', False),
            'duration': shot.get('duration', 0),
            'prep_start': prep_ts,
            'recording_start': rec_ts.get('start_time'),
            'recording_end': rec_ts.get('end_time'),
            'shot_end': last_phase.get('end_time') if last_phase else None,
        })
    return shots_info


# ── Geometry helpers ─────────────────────────────────────────────────────────

def compute_angle(a, b, c):
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _landmarks_to_array(landmarks):
    """Convert MediaPipe NormalizedLandmark list to (33, 4) numpy array."""
    arr = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        arr[i] = [lm.x, lm.y, lm.z, lm.visibility]
    return arr


# ── Pose estimation ──────────────────────────────────────────────────────────

def run_pose_estimation(video_path, shot_times):
    """Run MediaPipe pose on the full video and organise landmarks per shot.

    Returns {shot_number: {'recording': {frame_idx: (33,4) array}, 'full_window': {...}}}
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {total_frames} frames @ {fps:.1f} fps")

    model_path = str(MODEL_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pose model not found at {MODEL_PATH}. Download it with:\n"
            "  curl -L -o analysis/pose_landmarker_heavy.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_landmarks = {}

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(frame_idx * 1000 / fps)

            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                all_landmarks[frame_idx] = _landmarks_to_array(
                    results.pose_landmarks[0]
                )

            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"    Processed {frame_idx}/{total_frames} frames "
                      f"({frame_idx / total_frames * 100:.0f}%)")

    cap.release()
    print(f"    Done. Pose detected in {len(all_landmarks)}/{frame_idx} frames "
          f"({len(all_landmarks) / max(frame_idx, 1) * 100:.0f}%)")

    results_by_shot = {}
    for shot_info in shot_times:
        sn = shot_info['shot_number']

        rec_start_s = shot_info.get('recording_start')
        rec_end_s = shot_info.get('recording_end')
        prep_start_s = shot_info.get('prep_start')
        shot_end_s = shot_info.get('shot_end')

        recording_landmarks = {}
        full_window_landmarks = {}

        if rec_start_s is not None and rec_end_s is not None:
            rec_start = int(rec_start_s * fps)
            rec_end = int(rec_end_s * fps)
            for fi in range(rec_start, rec_end + 1):
                if fi in all_landmarks:
                    recording_landmarks[fi] = all_landmarks[fi]

        t_min = prep_start_s if prep_start_s is not None else rec_start_s
        t_max = shot_end_s if shot_end_s is not None else rec_end_s
        if t_min is not None and t_max is not None:
            f_start = int(t_min * fps)
            f_end = int(t_max * fps)
            for fi in range(f_start, f_end + 1):
                if fi in all_landmarks:
                    full_window_landmarks[fi] = all_landmarks[fi]

        results_by_shot[sn] = {
            'recording': recording_landmarks,
            'full_window': full_window_landmarks,
        }

    return results_by_shot


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_pose_features(landmarks_sequence):
    """Compute biomechanical features from time-ordered pose landmarks.

    Returns dict with frames, timestamps, and feature arrays.
    """
    if not landmarks_sequence:
        return None

    sorted_frames = sorted(landmarks_sequence.keys())
    n = len(sorted_frames)

    elbow_angles = np.zeros(n)
    shoulder_angles = np.zeros(n)
    knee_angles = np.zeros(n)
    wrist_heights = np.zeros(n)
    com_y = np.zeros(n)
    lean_angles = np.zeros(n)

    for i, fi in enumerate(sorted_frames):
        lm = landmarks_sequence[fi]

        r_shoulder = lm[RIGHT_SHOULDER, :2]
        r_elbow = lm[RIGHT_ELBOW, :2]
        r_wrist = lm[RIGHT_WRIST, :2]
        r_hip = lm[RIGHT_HIP, :2]
        r_knee = lm[RIGHT_KNEE, :2]
        r_ankle = lm[RIGHT_ANKLE, :2]
        l_hip = lm[LEFT_HIP, :2]
        l_shoulder = lm[LEFT_SHOULDER, :2]

        elbow_angles[i] = compute_angle(r_shoulder, r_elbow, r_wrist)
        shoulder_angles[i] = compute_angle(r_hip, r_shoulder, r_elbow)
        knee_angles[i] = compute_angle(r_hip, r_knee, r_ankle)
        wrist_heights[i] = 1.0 - r_wrist[1]
        com_y[i] = 1.0 - (r_hip[1] + l_hip[1]) / 2.0

        mid_shoulder = (r_shoulder + l_shoulder) / 2.0
        mid_hip = (r_hip + l_hip) / 2.0
        vertical = np.array([0.0, -1.0])
        lean_angles[i] = compute_angle(
            mid_hip + vertical, mid_hip, mid_shoulder
        )

    release_frame = _estimate_release_frame(sorted_frames, wrist_heights)

    frames_arr = np.array(sorted_frames)
    return {
        'frames': frames_arr,
        'timestamps': frames_arr / VIDEO_FPS,
        'elbow_angle': elbow_angles,
        'shoulder_angle': shoulder_angles,
        'knee_angle': knee_angles,
        'wrist_height': wrist_heights,
        'center_of_mass_y': com_y,
        'body_lean_angle': lean_angles,
        'body_lean': lean_angles,
        'release_frame': release_frame,
    }


def _estimate_release_frame(sorted_frames, wrist_heights):
    if len(sorted_frames) < 3:
        return int(sorted_frames[-1]) if sorted_frames else None

    velocity = np.gradient(wrist_heights)
    height_threshold = np.percentile(wrist_heights, 75)
    high_mask = wrist_heights >= height_threshold

    if not np.any(high_mask):
        high_mask = np.ones(len(sorted_frames), dtype=bool)

    masked_velocity = np.where(high_mask, velocity, -np.inf)
    release_idx = int(np.argmax(masked_velocity))
    return int(sorted_frames[release_idx])


# ── Made vs missed comparison ────────────────────────────────────────────────

def compare_made_vs_missed(all_features, shot_times):
    outcome_map = {s['shot_number']: s['success'] for s in shot_times}
    feature_keys = [
        'elbow_angle', 'shoulder_angle', 'knee_angle',
        'wrist_height', 'center_of_mass_y', 'body_lean_angle',
    ]

    made_features = {k: [] for k in feature_keys}
    missed_features = {k: [] for k in feature_keys}

    for sn, feats in all_features.items():
        if feats is None:
            continue
        bucket = made_features if outcome_map.get(sn, False) else missed_features
        for k in feature_keys:
            bucket[k].append(feats[k])

    comparison = {}
    for k in feature_keys:
        made_arrs = made_features[k]
        missed_arrs = missed_features[k]
        comp = {'feature': k}

        for label, arrs in [('made', made_arrs), ('missed', missed_arrs)]:
            if arrs:
                peaks = [float(np.max(a)) for a in arrs]
                mins = [float(np.min(a)) for a in arrs]
                ranges = [float(np.max(a) - np.min(a)) for a in arrs]
                peak_times = [int(np.argmax(a)) for a in arrs]
                means_over_time = [float(np.mean(a)) for a in arrs]

                comp[f'{label}_peak_mean'] = float(np.mean(peaks))
                comp[f'{label}_peak_std'] = float(np.std(peaks))
                comp[f'{label}_min_mean'] = float(np.mean(mins))
                comp[f'{label}_range_mean'] = float(np.mean(ranges))
                comp[f'{label}_range_std'] = float(np.std(ranges))
                comp[f'{label}_peak_timing_mean'] = float(np.mean(peak_times))
                comp[f'{label}_mean_trajectory'] = float(np.mean(means_over_time))
                comp[f'{label}_n'] = len(arrs)

                max_len = max(len(a) for a in arrs)
                if max_len > 0:
                    interp_arrs = []
                    common_x = np.linspace(0, 1, max_len)
                    for a in arrs:
                        orig_x = np.linspace(0, 1, len(a))
                        interp_arrs.append(np.interp(common_x, orig_x, a))
                    comp[f'{label}_trajectory_mean'] = np.mean(interp_arrs, axis=0).tolist()
                    comp[f'{label}_trajectory_std'] = np.std(interp_arrs, axis=0).tolist()
            else:
                comp[f'{label}_n'] = 0

        comparison[k] = comp

    made_releases, missed_releases = [], []
    for sn, feats in all_features.items():
        if feats is None or feats.get('release_frame') is None:
            continue
        if outcome_map.get(sn, False):
            made_releases.append(feats['release_frame'])
        else:
            missed_releases.append(feats['release_frame'])

    comparison['release_frame'] = {
        'feature': 'release_frame',
        'made_values': made_releases,
        'missed_values': missed_releases,
        'made_n': len(made_releases),
        'missed_n': len(missed_releases),
    }

    return comparison


# ── Drawing ──────────────────────────────────────────────────────────────────

def draw_pose_on_frame(frame, landmarks, connections=True):
    """Draw pose skeleton on a BGR frame. landmarks is (33, 4) array."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    visibility = landmarks[:, 3]
    vis_threshold = np.median(visibility)

    if connections:
        for idx_a, idx_b in POSE_CONNECTIONS:
            if visibility[idx_a] < 0.3 or visibility[idx_b] < 0.3:
                continue
            pt_a = (int(landmarks[idx_a, 0] * w), int(landmarks[idx_a, 1] * h))
            pt_b = (int(landmarks[idx_b, 0] * w), int(landmarks[idx_b, 1] * h))
            avg_vis = (visibility[idx_a] + visibility[idx_b]) / 2.0
            color = (0, 200, 0) if avg_vis >= vis_threshold else (0, 100, 200)
            cv2.line(annotated, pt_a, pt_b, color, 2, cv2.LINE_AA)

    for i in range(33):
        if visibility[i] < 0.3:
            continue
        cx = int(landmarks[i, 0] * w)
        cy = int(landmarks[i, 1] * h)
        color = (0, 255, 0) if visibility[i] >= vis_threshold else (0, 0, 255)
        cv2.circle(annotated, (cx, cy), 4, color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


# ── Annotated key-frame saving ───────────────────────────────────────────────

def _save_annotated_keyframes(video_path, pose_results, all_features, shot_times, output_dir):
    keyframe_dir = output_dir / 'keyframes'
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    key_frame_indices = set()
    frame_to_shot = {}

    for shot_info in shot_times:
        sn = shot_info['shot_number']
        feats = all_features.get(sn)
        if feats is None:
            continue

        release = feats.get('release_frame')
        if release is not None:
            key_frame_indices.add(release)
            frame_to_shot[release] = (sn, 'release')

        frames_arr = feats['frames']
        if len(frames_arr) > 0:
            mid_frame = int(frames_arr[len(frames_arr) // 2])
            key_frame_indices.add(mid_frame)
            frame_to_shot[mid_frame] = (sn, 'mid')

    if not key_frame_indices:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.warn(f"Cannot reopen video for keyframe extraction: {video_path}")
        return []

    outcome_map = {s['shot_number']: s['success'] for s in shot_times}
    saved_paths = []
    max_frame = max(key_frame_indices)
    frame_idx = 0

    while cap.isOpened() and frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in key_frame_indices:
            sn, label = frame_to_shot.get(frame_idx, (0, 'unknown'))
            lm_source = pose_results.get(sn, {})
            lm_dict = lm_source.get('recording', lm_source.get('full_window', {}))
            landmarks = lm_dict.get(frame_idx)

            annotated = draw_pose_on_frame(frame, landmarks) if landmarks is not None else frame.copy()
            outcome_str = 'made' if outcome_map.get(sn, False) else 'missed'
            filename = f'shot{sn:02d}_{outcome_str}_{label}_f{frame_idx}.png'
            save_path = keyframe_dir / filename
            cv2.imwrite(str(save_path), annotated)
            saved_paths.append(save_path)

        frame_idx += 1

    cap.release()
    return saved_paths


# ── JSON serialisation helpers ────────────────────────────────────────────────

def _features_to_serialisable(all_features):
    out = {}
    for sn, feats in all_features.items():
        if feats is None:
            out[str(sn)] = None
            continue
        d = {}
        for k, v in feats.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                d[k] = v.item()
            else:
                d[k] = v
        out[str(sn)] = d
    return out


def _comparison_to_serialisable(comparison):
    out = {}
    for k, v in comparison.items():
        d = {}
        for kk, vv in v.items():
            if isinstance(vv, np.ndarray):
                d[kk] = vv.tolist()
            elif isinstance(vv, (np.integer, np.floating)):
                d[kk] = vv.item()
            else:
                d[kk] = vv
        out[k] = d
    return out


# ── Main coordinator ─────────────────────────────────────────────────────────

def run_analysis(video_path, data_path, output_dir=None):
    video_path = Path(video_path)
    data_path = Path(data_path)
    if output_dir is None:
        output_dir = Path(__file__).parent / 'video_output' / 'pose'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading session from {data_path} ...")
    session = load_session(data_path)
    shot_times = get_shot_times(session)
    n_made = sum(1 for s in shot_times if s['success'])
    n_missed = sum(1 for s in shot_times if not s['success'])
    print(f"  {len(shot_times)} shots ({n_made} made, {n_missed} missed)")

    print(f"\nRunning pose estimation on {video_path} ...")
    pose_results = run_pose_estimation(video_path, shot_times)

    print("\nExtracting biomechanical features per shot ...")
    all_features = {}
    for shot_info in shot_times:
        sn = shot_info['shot_number']
        shot_data = pose_results.get(sn, {})
        lm_seq = shot_data.get('recording', {})
        if not lm_seq:
            lm_seq = shot_data.get('full_window', {})
        feats = extract_pose_features(lm_seq)
        all_features[sn] = feats
        n_frames = len(feats['frames']) if feats else 0
        print(f"  Shot {sn}: {'OK' if feats else 'no pose data'} ({n_frames} frames)")

    print("\nComparing made vs missed ...")
    comparison = compare_made_vs_missed(all_features, shot_times)

    print("\nSaving annotated key frames ...")
    saved_frames = _save_annotated_keyframes(
        video_path, pose_results, all_features, shot_times, output_dir
    )
    print(f"  Saved {len(saved_frames)} key-frame images")

    results = {
        'video': str(video_path),
        'session_data': str(data_path),
        'n_shots': len(shot_times),
        'n_made': n_made,
        'n_missed': n_missed,
        'features_per_shot': _features_to_serialisable(all_features),
        'comparison': _comparison_to_serialisable(comparison),
        'keyframes': [str(p) for p in saved_frames],
    }

    json_path = output_dir / 'pose_features.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run pose estimation on free-throw video')
    parser.add_argument('video_file', help='Path to the video file')
    parser.add_argument('data_file', help='Path to the session JSON file')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: analysis/video_output/pose/)')
    args = parser.parse_args()
    run_analysis(args.video_file, args.data_file, args.output)
