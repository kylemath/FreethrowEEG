"""
FreethrowEEG Video Figure Generator
Generates publication-quality figures combining video, pose estimation, and EEG
data for basketball free-throw neuroscience analysis.

Figures 9–13 complement the EEG-only figures (1–8) from analyze_session.py.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from pathlib import Path
import argparse
import cv2

try:
    import mediapipe as mp
    _POSE_CONNECTIONS = [(c.start, c.end) for c in
                         mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS]
    HAS_MEDIAPIPE = True
except (ImportError, AttributeError):
    _POSE_CONNECTIONS = []
    HAS_MEDIAPIPE = False

# ── Constants ────────────────────────────────────────────────────────────────

COLOR_MADE = '#2ca02c'
COLOR_MISSED = '#d62728'
PHASE_ORDER = ['prep', 'preShot', 'recording', 'postShot', 'review']
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

SHOT_PALETTE = [
    '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2',
    '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
]


# ── Video / pose helpers ─────────────────────────────────────────────────────

def _open_video(video_path):
    """Open a cv2 VideoCapture, raising on failure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    return cap


def _read_frame_at_time(cap, time_sec, fps):
    """Seek to *time_sec* and return an RGB frame (or None)."""
    frame_idx = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _extract_recording_frames(video_path, shot_time, n_frames=8):
    """Return *n_frames* evenly spaced RGB frames from the recording phase."""
    cap = _open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    t_start = shot_time['recording_start']
    t_end = shot_time['recording_end']
    duration = t_end - t_start
    times = np.linspace(0, duration, n_frames)

    frames = []
    for t in times:
        frames.append(_read_frame_at_time(cap, t_start + t, fps))
    cap.release()
    return frames, times


def _draw_pose_on_frame(frame, landmarks, alpha=0.5):
    """Overlay semi-transparent pose skeleton onto *frame*.

    *landmarks* is a list of (x_norm, y_norm) tuples (or None per joint).
    """
    if landmarks is None:
        return frame

    overlay = frame.copy()
    h, w = frame.shape[:2]

    for ci, cj in _POSE_CONNECTIONS:
        if ci < len(landmarks) and cj < len(landmarks):
            pt1, pt2 = landmarks[ci], landmarks[cj]
            if pt1 is not None and pt2 is not None:
                cv2.line(overlay,
                         (int(pt1[0] * w), int(pt1[1] * h)),
                         (int(pt2[0] * w), int(pt2[1] * h)),
                         (0, 255, 255), 2)

    for pt in landmarks:
        if pt is not None:
            cv2.circle(overlay, (int(pt[0] * w), int(pt[1] * h)), 3,
                       (255, 0, 0), -1)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def _get_landmarks_for_shot(pose_features, shot_num, frame_idx):
    """Look up pose landmarks for a specific shot and frame index."""
    shot_data = pose_features.get(shot_num, pose_features.get(str(shot_num), {}))
    lm_seq = shot_data.get('landmarks', [])
    if frame_idx < len(lm_seq):
        return lm_seq[frame_idx]
    return None


def _video_dims(video_path):
    cap = _open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h, fps


def _shot_time_for(shot_times, shot_number):
    return next((s for s in shot_times if s['shot_number'] == shot_number), None)


# ── Figure 9: Shot Stop-Motion Montage ───────────────────────────────────────

def fig_shot_montage(session, shot_times, pose_features, video_path, fig_dir,
                     n_frames=8):
    """Stop-motion strip per shot with pose overlay and outcome-coloured borders."""
    shots = session['shots']
    made = [s for s in shots if s['success']]
    missed = [s for s in shots if not s['success']]
    ordered = made + missed
    n_shots = len(ordered)

    fig, axes = plt.subplots(n_shots, n_frames,
                             figsize=(n_frames * 2, n_shots * 2.2))
    fig.suptitle('Figure 9 — Shot Stop-Motion Montage', fontsize=14, y=1.0)

    if n_shots == 1:
        axes = axes.reshape(1, -1)

    for row, shot in enumerate(ordered):
        snum = shot['shotNumber']
        st = _shot_time_for(shot_times, snum)
        border = COLOR_MADE if shot['success'] else COLOR_MISSED
        tag = 'Made' if shot['success'] else 'Missed'

        if st is None:
            for c in range(n_frames):
                axes[row, c].axis('off')
            continue

        frames, times = _extract_recording_frames(video_path, st, n_frames)

        for col, (frame, t) in enumerate(zip(frames, times)):
            ax = axes[row, col]
            if frame is not None:
                lm = _get_landmarks_for_shot(pose_features, snum, col)
                ax.imshow(_draw_pose_on_frame(frame, lm, alpha=0.4))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(3)
            ax.set_xlabel(f't={t:.1f}s', fontsize=7, labelpad=1)
            if col == 0:
                ax.set_ylabel(f'S{snum} ({tag})', fontsize=8, rotation=0,
                              labelpad=45, va='center', color=border,
                              fontweight='bold')

    plt.tight_layout(rect=[0.06, 0, 1, 0.97])
    path = fig_dir / 'fig_shot_montage.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure 10: Ghost Shot Overlay ────────────────────────────────────────────

def _release_frame(video_path, shot_time, fps):
    """Read frame at estimated release (midpoint of recording)."""
    cap = _open_video(video_path)
    t = (shot_time['recording_start'] + shot_time['recording_end']) / 2
    frame = _read_frame_at_time(cap, t, fps)
    cap.release()
    return frame


def fig_ghost_overlay(session, shot_times, pose_features, video_path, fig_dir):
    """Ghost overlay of release frames for made vs missed, plus skeleton panel."""
    shots = session['shots']
    made = [s for s in shots if s['success']]
    missed = [s for s in shots if not s['success']]
    w, h, fps = _video_dims(video_path)

    def _composite(group):
        acc = np.zeros((h, w, 3), dtype=np.float64)
        n = 0
        for shot in group:
            st = _shot_time_for(shot_times, shot['shotNumber'])
            if st is None:
                continue
            frame = _release_frame(video_path, st, fps)
            if frame is not None:
                acc += frame.astype(np.float64)
                n += 1
        return np.clip(acc / max(n, 1), 0, 255).astype(np.uint8), n

    made_ghost, n_m = _composite(made)
    missed_ghost, n_mi = _composite(missed)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.15)
    ax_mg = fig.add_subplot(gs[0, 0])
    ax_mig = fig.add_subplot(gs[0, 1])
    ax_sm = fig.add_subplot(gs[1, 0])
    ax_smi = fig.add_subplot(gs[1, 1])
    fig.suptitle('Figure 10 — Ghost Shot Overlay at Release', fontsize=14,
                 y=0.98)

    ax_mg.imshow(made_ghost)
    ax_mg.set_title(f'Made Shots (n={len(made)})', fontsize=11,
                    color=COLOR_MADE, fontweight='bold')
    ax_mg.axis('off')

    ax_mig.imshow(missed_ghost)
    ax_mig.set_title(f'Missed Shots (n={len(missed)})', fontsize=11,
                     color=COLOR_MISSED, fontweight='bold')
    ax_mig.axis('off')

    for ax, group, tc, lab in [
        (ax_sm, made, COLOR_MADE, 'Made'),
        (ax_smi, missed, COLOR_MISSED, 'Missed'),
    ]:
        ax.set_facecolor('black')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        for idx, shot in enumerate(group):
            snum = shot['shotNumber']
            sdata = pose_features.get(snum, pose_features.get(str(snum), {}))

            # Handle both formats: list-of-landmarks or raw_landmarks dict
            lm = None
            if 'raw_landmarks' in sdata:
                rl = sdata['raw_landmarks']
                if rl:
                    mid_key = sorted(rl.keys())[len(rl) // 2]
                    lm_arr = np.asarray(rl[mid_key])
                    lm = [(lm_arr[j, 0], lm_arr[j, 1]) if lm_arr[j, 3] > 0.3 else None
                          for j in range(lm_arr.shape[0])]
            elif 'landmarks' in sdata:
                lm_seq = sdata['landmarks']
                if lm_seq:
                    lm = lm_seq[len(lm_seq) // 2]

            if lm is None:
                continue
            color = SHOT_PALETTE[idx % len(SHOT_PALETTE)]
            for ci, cj in _POSE_CONNECTIONS:
                if (ci < len(lm) and cj < len(lm)
                        and lm[ci] is not None and lm[cj] is not None):
                    ax.plot([lm[ci][0] * w, lm[cj][0] * w],
                            [lm[ci][1] * h, lm[cj][1] * h],
                            color=color, alpha=0.6, linewidth=1.5)
            for pt in lm:
                if pt is not None:
                    ax.plot(pt[0] * w, pt[1] * h, 'o', color=color,
                            markersize=3, alpha=0.7)

        ax.set_title(f'{lab} – Pose Skeletons', fontsize=10, color=tc)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = fig_dir / 'fig_ghost_overlay.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure 11: Pose Kinematics Trajectories ─────────────────────────────────

def fig_pose_kinematics(session, shot_times, pose_features, fig_dir):
    """Time-series of biomechanical features during recording, made vs missed."""
    features = [
        ('elbow_angle', 'Elbow Angle (°)'),
        ('wrist_height', 'Wrist Height (norm.)'),
        ('knee_angle', 'Knee Angle (°)'),
        ('body_lean', 'Body Lean (°)'),
    ]

    shots = session['shots']
    made_nums = {s['shotNumber'] for s in shots if s['success']}
    missed_nums = {s['shotNumber'] for s in shots if not s['success']}

    durations = [st['recording_end'] - st['recording_start'] for st in shot_times]
    max_dur = max(durations) if durations else 3.0
    common_t = np.linspace(0, max_dur, 100)
    t_release = max_dur * 0.6

    fig, axes = plt.subplots(len(features), 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Figure 11 — Pose Kinematics During Shot Execution',
                 fontsize=14, y=0.98)

    for ax, (feat_key, feat_label) in zip(axes, features):
        made_traces, missed_traces = [], []

        for snum_raw, sdata in pose_features.items():
            snum = int(snum_raw) if isinstance(snum_raw, str) else snum_raw
            ts_raw = np.asarray(sdata.get('timestamps', []), dtype=float)
            vals_raw = np.asarray(sdata.get(feat_key, []), dtype=float)

            if ts_raw.size == 0 or vals_raw.size == 0:
                continue
            n = min(len(ts_raw), len(vals_raw))
            ts_raw, vals_raw = ts_raw[:n], vals_raw[:n]

            interp_v = np.interp(common_t, ts_raw, vals_raw,
                                 left=np.nan, right=np.nan)

            if snum in made_nums:
                made_traces.append(interp_v)
                ax.plot(common_t, interp_v, color=COLOR_MADE,
                        linewidth=0.6, alpha=0.35)
            elif snum in missed_nums:
                missed_traces.append(interp_v)
                ax.plot(common_t, interp_v, color=COLOR_MISSED,
                        linewidth=0.6, alpha=0.35)

        for traces, color, label in [
            (made_traces, COLOR_MADE, f'Made (n={len(made_traces)})'),
            (missed_traces, COLOR_MISSED, f'Missed (n={len(missed_traces)})'),
        ]:
            if not traces:
                continue
            mat = np.array(traces)
            mean_v = np.nanmean(mat, axis=0)
            valid_n = np.sum(~np.isnan(mat), axis=0).clip(min=1)
            sem_v = np.nanstd(mat, axis=0) / np.sqrt(valid_n)
            ax.plot(common_t, mean_v, color=color, linewidth=2.5, label=label)
            ax.fill_between(common_t, mean_v - sem_v, mean_v + sem_v,
                            color=color, alpha=0.15)

        ax.axvline(t_release, color='gray', linewidth=1.2, linestyle='--',
                    alpha=0.7)
        ax.set_ylabel(feat_label, fontsize=9)
        ax.tick_params(labelsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc='upper right')

    axes[0].annotate('Est. release', xy=(t_release, 1), xycoords=('data', 'axes fraction'),
                     fontsize=7, ha='center', va='bottom', color='gray')
    axes[-1].set_xlabel('Time from recording start (s)', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = fig_dir / 'fig_pose_kinematics.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure 12: Pose + EEG Combined ──────────────────────────────────────────

def _extract_recording_eeg(shot, band):
    """Return (time_from_rec_start, power) arrays for the recording phase."""
    rec = shot.get('eegData', {}).get('recording', {}).get(band, [])
    if not rec:
        return None, None
    timestamps = np.array([e['timestamp'] for e in rec])
    power = np.array([e['power'] for e in rec])
    return timestamps - timestamps[0], power


def fig_pose_eeg_combined(session, shot_times, pose_features, fig_dir):
    """Multi-panel figure aligning pose kinematics with EEG during execution."""
    shots = session['shots']
    made_nums = {s['shotNumber'] for s in shots if s['success']}
    missed_nums = {s['shotNumber'] for s in shots if not s['success']}

    durations = [st['recording_end'] - st['recording_start'] for st in shot_times]
    max_dur = max(durations) if durations else 3.0
    common_t = np.linspace(0, max_dur, 100)

    panels = [
        ('elbow_angle',       'Elbow Angle (°)',   'pose'),
        ('wrist_height',      'Wrist Height (norm.)', 'pose'),
        ('alpha',             'Alpha Power (μV)',  'eeg'),
        ('theta_alpha_ratio', 'θ/α Ratio',        'eeg'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle('Figure 12 — Pose Kinematics + EEG During Execution',
                 fontsize=14, y=0.98)
    flat = axes.flatten()

    corr_results = []

    for ax, (key, label, dtype) in zip(flat, panels):
        made_traces, missed_traces = [], []

        if dtype == 'pose':
            for snum_raw, sdata in pose_features.items():
                snum = int(snum_raw) if isinstance(snum_raw, str) else snum_raw
                ts_r = np.asarray(sdata.get('timestamps', []), dtype=float)
                vr = np.asarray(sdata.get(key, []), dtype=float)
                if ts_r.size == 0 or vr.size == 0:
                    continue
                n = min(len(ts_r), len(vr))
                iv = np.interp(common_t, ts_r[:n], vr[:n],
                               left=np.nan, right=np.nan)
                (made_traces if snum in made_nums else missed_traces).append(iv)
        else:
            for shot in shots:
                if key == 'theta_alpha_ratio':
                    t_th, p_th = _extract_recording_eeg(shot, 'theta')
                    t_al, p_al = _extract_recording_eeg(shot, 'alpha')
                    if t_th is None or t_al is None:
                        continue
                    iv = (np.interp(common_t, t_th, p_th, left=np.nan, right=np.nan)
                          / np.clip(np.interp(common_t, t_al, p_al,
                                              left=np.nan, right=np.nan), 1e-6, None))
                else:
                    t_eeg, p_eeg = _extract_recording_eeg(shot, key)
                    if t_eeg is None:
                        continue
                    iv = np.interp(common_t, t_eeg, p_eeg,
                                   left=np.nan, right=np.nan)
                (made_traces if shot['success'] else missed_traces).append(iv)

        for traces, color, lbl in [
            (made_traces, COLOR_MADE, f'Made (n={len(made_traces)})'),
            (missed_traces, COLOR_MISSED, f'Missed (n={len(missed_traces)})'),
        ]:
            if not traces:
                continue
            mat = np.array(traces)
            mean_v = np.nanmean(mat, axis=0)
            valid_n = np.sum(~np.isnan(mat), axis=0).clip(min=1)
            sem_v = np.nanstd(mat, axis=0) / np.sqrt(valid_n)
            ax.plot(common_t, mean_v, color=color, linewidth=2, label=lbl)
            ax.fill_between(common_t, mean_v - sem_v, mean_v + sem_v,
                            color=color, alpha=0.15)

        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='upper right')

    # Cross-domain correlations (shot-level mean pose vs mean EEG)
    for pk in ('elbow_angle', 'wrist_height'):
        pose_vals, alpha_vals = [], []
        for shot in shots:
            snum = shot['shotNumber']
            sd = pose_features.get(snum, pose_features.get(str(snum), {}))
            pv = sd.get(pk, [])
            if not pv:
                continue
            t_eeg, p_eeg = _extract_recording_eeg(shot, 'alpha')
            if t_eeg is None:
                continue
            pose_vals.append(np.nanmean(pv))
            alpha_vals.append(np.nanmean(p_eeg))
        if len(pose_vals) >= 4:
            r, p = pearsonr(pose_vals, alpha_vals)
            corr_results.append((pk, 'alpha', r, p))

    if corr_results:
        parts = []
        for pk, ek, r, p in corr_results:
            sig = '*' if p < 0.05 else ''
            parts.append(f'r({pk}, {ek})={r:.2f}, p={p:.3f}{sig}')
        fig.text(0.5, 0.005, '  |  '.join(parts), ha='center', fontsize=8,
                 style='italic', color='#555555')

    for a in axes[1]:
        a.set_xlabel('Time from recording start (s)', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = fig_dir / 'fig_pose_eeg_combined.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure 13: Average Frame Composite ──────────────────────────────────────

def fig_average_frames(session, shot_times, video_path, fig_dir):
    """Pixel-averaged frames at start / midpoint / end for made vs missed,
    plus absolute-difference row."""
    shots = session['shots']
    made = [s for s in shots if s['success']]
    missed = [s for s in shots if not s['success']]
    w, h, fps = _video_dims(video_path)

    labels = ['Start', 'Midpoint', 'End']
    fracs = [0.0, 0.5, 1.0]

    def _avg_frame(group, frac):
        acc = np.zeros((h, w, 3), dtype=np.float64)
        n = 0
        for shot in group:
            st = _shot_time_for(shot_times, shot['shotNumber'])
            if st is None:
                continue
            t = st['recording_start'] + frac * (st['recording_end'] - st['recording_start'])
            cap = _open_video(video_path)
            frame = _read_frame_at_time(cap, t, fps)
            cap.release()
            if frame is not None:
                acc += frame.astype(np.float64)
                n += 1
        return (acc / max(n, 1)).astype(np.uint8), n

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Figure 13 — Average Frame Composites: Made vs Missed',
                 fontsize=14, y=0.98)

    for col, (lbl, frac) in enumerate(zip(labels, fracs)):
        avg_m, _ = _avg_frame(made, frac)
        avg_mi, _ = _avg_frame(missed, frac)

        axes[0, col].imshow(avg_m)
        axes[0, col].set_title(f'Made – {lbl}', fontsize=10, color=COLOR_MADE)
        axes[0, col].axis('off')

        axes[1, col].imshow(avg_mi)
        axes[1, col].set_title(f'Missed – {lbl}', fontsize=10, color=COLOR_MISSED)
        axes[1, col].axis('off')

        diff = np.abs(avg_m.astype(np.float64) - avg_mi.astype(np.float64))
        dmax = diff.max()
        diff_img = (diff / dmax * 255).astype(np.uint8) if dmax > 0 else diff.astype(np.uint8)
        axes[2, col].imshow(diff_img)
        axes[2, col].set_title(f'|Difference| – {lbl}', fontsize=10,
                               color='#555555')
        axes[2, col].axis('off')

    for row, rlabel in enumerate([f'Made (n={len(made)})',
                                   f'Missed (n={len(missed)})',
                                   'Abs. Difference']):
        axes[row, 0].annotate(rlabel, xy=(-0.15, 0.5),
                              xycoords='axes fraction', fontsize=10,
                              rotation=90, va='center', ha='center',
                              fontweight='bold')

    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    path = fig_dir / 'fig_average_frames.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Main runner ──────────────────────────────────────────────────────────────

def generate_all_figures(session, shot_times, pose_features, video_path,
                         fig_dir):
    """Generate all video-based figures (9–13).

    Returns a dict mapping figure names to saved file paths.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)

    figures = {}

    print("Generating video-based figures...")

    print("  [9/13] Shot stop-motion montage...")
    figures['shot_montage'] = fig_shot_montage(
        session, shot_times, pose_features, video_path, fig_dir)

    print("  [10/13] Ghost shot overlay...")
    figures['ghost_overlay'] = fig_ghost_overlay(
        session, shot_times, pose_features, video_path, fig_dir)

    print("  [11/13] Pose kinematics trajectories...")
    figures['pose_kinematics'] = fig_pose_kinematics(
        session, shot_times, pose_features, fig_dir)

    print("  [12/13] Pose + EEG combined...")
    figures['pose_eeg_combined'] = fig_pose_eeg_combined(
        session, shot_times, pose_features, fig_dir)

    print("  [13/13] Average frame composites...")
    figures['average_frames'] = fig_average_frames(
        session, shot_times, video_path, fig_dir)

    print(f"\nAll video figures saved to {fig_dir}/")
    for name, fpath in figures.items():
        print(f"  {name}: {fpath}")

    return figures


# ── CLI ──────────────────────────────────────────────────────────────────────

def _load_data(data_path):
    """Load session JSON and extract shot_times / pose_features.

    Accepts either a combined JSON (with top-level keys 'session',
    'shot_times', 'pose_features') or a plain session JSON.  When
    shot_times are absent they are synthesized from shot metadata.
    """
    with open(data_path) as f:
        data = json.load(f)

    session = data if 'shots' in data else data.get('session', data)

    shot_times = data.get('shot_times', [])
    if not shot_times:
        for s in session.get('shots', []):
            prep = s.get('prep_start', 0)
            shot_times.append({
                'shot_number': s['shotNumber'],
                'success': s['success'],
                'prep_start': prep,
                'recording_start': s.get('recording_start', prep + 15),
                'recording_end': s.get('recording_end', prep + 18),
                'shot_end': s.get('shot_end', prep + 23),
            })

    pose_features = data.get('pose_features', {})
    pose_features = {
        (int(k) if isinstance(k, str) and k.isdigit() else k): v
        for k, v in pose_features.items()
    }

    return session, shot_times, pose_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate video-based figures for FreethrowEEG analysis')
    parser.add_argument('data_file', help='Path to session / analysis JSON')
    parser.add_argument('video_file', help='Path to session video (.mp4)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: analysis/figures/)')
    args = parser.parse_args()

    session, shot_times, pose_features = _load_data(args.data_file)

    fig_dir = Path(args.output) if args.output else Path(__file__).parent / 'figures'

    generate_all_figures(session, shot_times, pose_features,
                         args.video_file, fig_dir)
