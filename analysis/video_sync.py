"""
FreethrowEEG Video Synchronization
Synchronizes video with session JSON data, extracts per-shot clips,
key frames, and film-strip montages.
"""

import json
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PHASE_ORDER = ['prep', 'preShot', 'recording', 'postShot', 'review']
CLIP_PAD_S = 2.0


def load_session(filepath):
    with open(filepath) as f:
        return json.load(f)


def _phase_timestamps(shot, phase):
    """Return all timestamps from any band in a given phase, sorted."""
    phase_data = shot['eegData'].get(phase)
    if not phase_data:
        return []
    for band in phase_data:
        entries = phase_data[band]
        if entries:
            return sorted(e['timestamp'] for e in entries)
    return []


# ── 1. Shot timing extraction ───────────────────────────────────────────────

def get_shot_times(session):
    """Extract timing info for every shot from the session JSON.

    Returns a list of dicts with keys:
        shot_number, success, prep_start, recording_start, recording_end,
        shot_end, duration
    """
    shot_times = []
    for shot in session['shots']:
        info = {
            'shot_number': shot['shotNumber'],
            'success': shot['success'],
            'duration': shot['duration'],
            'prep_start': None,
            'recording_start': None,
            'recording_end': None,
            'shot_end': None,
        }

        prep_ts = _phase_timestamps(shot, 'prep')
        if prep_ts:
            info['prep_start'] = prep_ts[0]

        rec_ts = _phase_timestamps(shot, 'recording')
        if rec_ts:
            info['recording_start'] = rec_ts[0]
            info['recording_end'] = rec_ts[-1]

        for phase in reversed(PHASE_ORDER):
            ts = _phase_timestamps(shot, phase)
            if ts:
                info['shot_end'] = ts[-1]
                break

        shot_times.append(info)

    return shot_times


# ── 2. Extract individual shot clips ────────────────────────────────────────

def _read_frame_at(cap, time_s, fps):
    """Seek to *time_s* and return the frame (or None)."""
    frame_idx = int(round(time_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def _write_clip(cap, fps, width, height, start_s, end_s, out_path):
    """Write frames between start_s and end_s to an mp4 file."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    start_frame = max(0, int(round(start_s * fps)))
    end_frame = int(round(end_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    writer.release()


def extract_shot_clips(video_path, shot_times, output_dir):
    """Cut individual clips and combined made/missed compilations."""
    clips_dir = Path(output_dir) / 'clips'
    clips_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    print(f"  Video: {width}x{height} @ {fps:.1f} fps, {video_duration:.1f}s")

    clip_paths = {'made': [], 'missed': []}

    for st in shot_times:
        if st['prep_start'] is None or st['shot_end'] is None:
            print(f"  Shot {st['shot_number']}: skipped (missing phase data)")
            continue

        start_s = max(0.0, st['prep_start'] - CLIP_PAD_S)
        end_s = min(video_duration, st['shot_end'] + CLIP_PAD_S)

        label = 'made' if st['success'] else 'missed'
        fname = f"shot_{st['shot_number']:02d}_{label}.mp4"
        out_path = clips_dir / fname

        _write_clip(cap, fps, width, height, start_s, end_s, out_path)
        clip_paths[label].append(out_path)
        print(f"  Shot {st['shot_number']:2d} ({label:6s}): "
              f"{start_s:.1f}s – {end_s:.1f}s -> {fname}")

    for group in ('made', 'missed'):
        if not clip_paths[group]:
            continue
        combined_path = clips_dir / f"all_{group}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(combined_path), fourcc, fps, (width, height))
        for cp in clip_paths[group]:
            src = cv2.VideoCapture(str(cp))
            while True:
                ret, frame = src.read()
                if not ret:
                    break
                writer.write(frame)
            src.release()
        writer.release()
        print(f"  Combined {group}: {combined_path.name} "
              f"({len(clip_paths[group])} clips)")

    cap.release()
    return clips_dir


# ── 3. Extract still frames and montages ────────────────────────────────────

def extract_still_frames(video_path, shot_times, output_dir, n_frames=8):
    """Extract evenly-spaced frames from each recording phase plus key moments."""
    frames_dir = Path(output_dir) / 'frames'
    montages_dir = Path(output_dir) / 'montages'
    frames_dir.mkdir(parents=True, exist_ok=True)
    montages_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    for st in shot_times:
        sn = st['shot_number']

        if st['recording_start'] is None or st['recording_end'] is None:
            print(f"  Shot {sn}: skipped frames (no recording phase)")
            continue

        # --- Evenly spaced frames across the recording phase ---
        rec_start = st['recording_start']
        rec_end = st['recording_end']
        rec_times = np.linspace(rec_start, rec_end, n_frames)

        rec_frames = []
        rec_labels = []
        for i, t in enumerate(rec_times):
            if t > video_duration:
                continue
            frame = _read_frame_at(cap, t, fps)
            if frame is not None:
                fname = f"shot_{sn:02d}_rec_{i:02d}.png"
                cv2.imwrite(str(frames_dir / fname), frame)
                rec_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                rec_labels.append(f"{t:.2f}s")

        if rec_frames:
            label = 'made' if st['success'] else 'missed'
            title = f"Shot {sn} ({label}) — Recording Phase"
            montage_path = montages_dir / f"shot_{sn:02d}_montage.png"
            create_filmstrip(rec_frames, rec_labels, montage_path, title)

        # --- Key moment frames ---
        key_moments = _build_key_moments(st, video_duration)
        key_frames = []
        key_labels = []
        for name, t in key_moments:
            frame = _read_frame_at(cap, t, fps)
            if frame is not None:
                fname = f"shot_{sn:02d}_key_{name}.png"
                cv2.imwrite(str(frames_dir / fname), frame)
                key_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                key_labels.append(name)

        if key_frames:
            label = 'made' if st['success'] else 'missed'
            title = f"Shot {sn} ({label}) — Key Moments"
            key_path = montages_dir / f"shot_{sn:02d}_key_moments.png"
            create_filmstrip(key_frames, key_labels, key_path, title)

        print(f"  Shot {sn:2d}: {len(rec_frames)} rec frames, "
              f"{len(key_frames)} key frames")

    cap.release()
    return frames_dir, montages_dir


def _build_key_moments(st, video_duration):
    """Return list of (label, time_s) tuples for key moments in a shot."""
    moments = []

    def _add(name, t):
        if t is not None and 0 <= t <= video_duration:
            moments.append((name, t))

    _add('prep_start', st.get('prep_start'))

    preshot_start = st.get('recording_start')
    if preshot_start is not None and st.get('prep_start') is not None:
        _add('preshot_start', st['prep_start'] + 5.0)

    _add('throw_start', st.get('recording_start'))

    rec_s = st.get('recording_start')
    rec_e = st.get('recording_end')
    if rec_s is not None and rec_e is not None:
        _add('throw_mid', (rec_s + rec_e) / 2.0)

    _add('throw_end', st.get('recording_end'))

    if st.get('recording_end') is not None:
        _add('postshot_start', st['recording_end'] + 0.25)

    return moments


# ── 4. Film-strip montage creation ──────────────────────────────────────────

def create_filmstrip(frames, labels, output_path, title=""):
    """Arrange frames horizontally as a film-strip montage with labels."""
    n = len(frames)
    if n == 0:
        return

    thumb_h = 240
    aspect = frames[0].shape[1] / frames[0].shape[0]
    thumb_w = int(thumb_h * aspect)

    fig_w = max(6, n * 2.2)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.5))
    if n == 1:
        axes = [axes]

    for ax, frame, lbl in zip(axes, frames, labels):
        resized = cv2.resize(frame, (thumb_w, thumb_h),
                             interpolation=cv2.INTER_AREA)
        ax.imshow(resized)
        ax.set_title(lbl, fontsize=7, pad=2)
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=10, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


# ── 5. Main coordinator ────────────────────────────────────────────────────

def run_sync(data_path, video_path, output_dir=None):
    """Load session, extract shot times, clips, frames, and montages."""
    data_path = Path(data_path)
    video_path = Path(video_path)

    if output_dir is None:
        output_dir = Path(__file__).parent / 'video_output'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading session from {data_path}")
    session = load_session(data_path)
    print(f"  Player: {session['playerName']}")
    print(f"  Shots: {session['totalShots']}  "
          f"({sum(1 for s in session['shots'] if s['success'])} made, "
          f"{sum(1 for s in session['shots'] if not s['success'])} missed)")
    print(f"  Session duration: {session['sessionDuration']:.1f}s")

    print("\nExtracting shot times...")
    shot_times = get_shot_times(session)
    for st in shot_times:
        label = 'made' if st['success'] else 'missed'
        print(f"  Shot {st['shot_number']:2d} ({label:6s}): "
              f"prep={st['prep_start']:.1f}s  "
              f"rec={st['recording_start']:.1f}–{st['recording_end']:.1f}s  "
              f"end={st['shot_end']:.1f}s")

    print("\nExtracting clips...")
    extract_shot_clips(video_path, shot_times, output_dir)

    print("\nExtracting frames & montages...")
    extract_still_frames(video_path, shot_times, output_dir)

    times_path = output_dir / 'shot_times.json'
    with open(times_path, 'w') as f:
        json.dump(shot_times, f, indent=2)
    print(f"\nShot times saved to {times_path}")

    print("Done.")
    return shot_times


# ── 6. CLI interface ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Synchronize video with FreethrowEEG session data')
    parser.add_argument('data_file',
                        help='Path to session JSON file')
    parser.add_argument('video_file',
                        help='Path to session video file (.webm or .mp4)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: analysis/video_output/)')
    args = parser.parse_args()
    run_sync(args.data_file, args.video_file, args.output)


if __name__ == '__main__':
    main()
