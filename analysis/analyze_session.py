"""
FreethrowEEG Session Analysis
Loads session JSON, performs signal processing and statistical analysis,
generates publication-quality figures and summary statistics.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt, welch
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr
from pathlib import Path
import argparse
import sys

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_RANGES = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 50)
}
BAND_COLORS = {
    'delta': '#1f77b4', 'theta': '#ff7f0e', 'alpha': '#2ca02c',
    'beta': '#d62728', 'gamma': '#9467bd'
}
PHASE_ORDER = ['prep', 'preShot', 'recording', 'postShot', 'review']
PHASE_LABELS = {
    'prep': 'Preparation', 'preShot': 'Pre-Shot', 'recording': 'Execution',
    'postShot': 'Post-Shot', 'review': 'Review'
}


def load_session(filepath):
    with open(filepath) as f:
        return json.load(f)


def butter_filter(data, fs, cutoff, btype, order=4):
    nyq = 0.5 * fs
    if btype == 'band':
        low, high = cutoff
        b, a = butter(order, [low / nyq, high / nyq], btype='bandpass')
    elif btype == 'low':
        b, a = butter(order, cutoff / nyq, btype='low')
    elif btype == 'high':
        b, a = butter(order, cutoff / nyq, btype='high')
    else:
        raise ValueError(f"Unknown filter type: {btype}")
    return filtfilt(b, a, data)


def estimate_fs(timestamps):
    diffs = np.diff(timestamps)
    return 1.0 / np.median(diffs)


# ── Figure 1: Continuous band power with shot markers ────────────────────────

def fig_continuous_power(session, fig_dir):
    """Full-session band power time series with shot onset markers."""
    ts = np.array(session['eegData']['timestamps'])
    bands_data = session['eegData']['bands']

    fig, axes = plt.subplots(len(BANDS), 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Continuous Band Power Over Session', fontsize=14, y=0.98)

    for ax, band in zip(axes, BANDS):
        power = np.array(bands_data[band])
        ax.plot(ts, power, color=BAND_COLORS[band], linewidth=0.6, alpha=0.85)
        ax.set_ylabel(f'{band.capitalize()}\n(μV)', fontsize=9)
        ax.tick_params(labelsize=8)

        for shot in session['shots']:
            t_shot = shot['duration']
            color = '#2ca02c' if shot['success'] else '#d62728'
            ax.axvline(t_shot, color=color, linewidth=1.2, alpha=0.7, linestyle='--')

        ax.set_xlim(ts[0], ts[-1])

    axes[-1].set_xlabel('Time (s)', fontsize=10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2ca02c', linestyle='--', label='Made'),
        Line2D([0], [0], color='#d62728', linestyle='--', label='Missed'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = fig_dir / 'fig_continuous_power.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure 2: Raw signal with filtering demonstrations ───────────────────────

def fig_raw_filtering(session, fig_dir):
    """Show raw band signals with low-pass, high-pass, and bandpass filtering."""
    ts = np.array(session['eegData']['timestamps'])
    raw_bands = session['eegData']['rawBands']
    fs = estimate_fs(ts)

    window_start_idx = 0
    window_end_idx = min(len(ts), int(60 * fs))

    ts_win = ts[window_start_idx:window_end_idx]
    raw_delta = np.array(raw_bands['delta'][window_start_idx:window_end_idx])

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Signal Filtering Demonstration (Delta Band, First 60s)', fontsize=13, y=0.98)

    axes[0].plot(ts_win, raw_delta, 'k', linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel('Raw (μV)', fontsize=9)
    axes[0].set_title('Unfiltered Raw Signal', fontsize=10)

    lp = butter_filter(raw_delta, fs, 1.0, 'low', order=2)
    axes[1].plot(ts_win, lp, color='#1f77b4', linewidth=0.8)
    axes[1].set_ylabel('μV', fontsize=9)
    axes[1].set_title('Low-Pass Filtered (< 1 Hz)', fontsize=10)

    hp = butter_filter(raw_delta, fs, 0.5, 'high', order=2)
    axes[2].plot(ts_win, hp, color='#ff7f0e', linewidth=0.5)
    axes[2].set_ylabel('μV', fontsize=9)
    axes[2].set_title('High-Pass Filtered (> 0.5 Hz)', fontsize=10)

    bp = butter_filter(raw_delta, fs, (0.5, 1.5), 'band', order=2)
    axes[3].plot(ts_win, bp, color='#2ca02c', linewidth=0.8)
    axes[3].set_ylabel('μV', fontsize=9)
    axes[3].set_title('Bandpass Filtered (0.5 – 1.5 Hz)', fontsize=10)

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    for ax in axes:
        ax.tick_params(labelsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = fig_dir / 'fig_raw_filtering.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure 3: Shot-locked averages (all shots, per band) ────────────────────

def _extract_shot_epoch(shot, band, align_to_recording=True):
    """Extract a time-aligned epoch for one shot/band across all phases."""
    phase_data = {}
    for phase in PHASE_ORDER:
        if phase not in shot['eegData']:
            continue
        entries = shot['eegData'][phase].get(band, [])
        if entries:
            phase_data[phase] = {
                'timestamps': np.array([e['timestamp'] for e in entries]),
                'power': np.array([e['power'] for e in entries]),
            }
    if 'recording' not in phase_data:
        return None, None

    t0 = phase_data['recording']['timestamps'][0]
    all_t, all_p = [], []
    for phase in PHASE_ORDER:
        if phase in phase_data:
            all_t.append(phase_data[phase]['timestamps'] - t0)
            all_p.append(phase_data[phase]['power'])
    return np.concatenate(all_t), np.concatenate(all_p)


def fig_shot_locked_average(session, fig_dir):
    """ERP-style shot-locked averages for each band, all shots combined."""
    fig, axes = plt.subplots(len(BANDS), 1, figsize=(12, 12), sharex=True)
    fig.suptitle('Shot-Locked Band Power Averages (All Shots)', fontsize=14, y=0.98)

    common_t = np.arange(-12, 8, 0.25)

    for ax, band in zip(axes, BANDS):
        all_interp = []
        for shot in session['shots']:
            t_epoch, p_epoch = _extract_shot_epoch(shot, band)
            if t_epoch is None:
                continue
            interp_p = np.interp(common_t, t_epoch, p_epoch,
                                 left=np.nan, right=np.nan)
            all_interp.append(interp_p)

        if not all_interp:
            continue
        mat = np.array(all_interp)
        mean_p = np.nanmean(mat, axis=0)
        sem_p = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))

        ax.plot(common_t, mean_p, color=BAND_COLORS[band], linewidth=1.5)
        ax.fill_between(common_t, mean_p - sem_p, mean_p + sem_p,
                        color=BAND_COLORS[band], alpha=0.2)
        ax.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.7)
        ax.set_ylabel(f'{band.capitalize()}\n(μV)', fontsize=9)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel('Time relative to shot execution (s)', fontsize=10)
    axes[0].annotate('Shot', xy=(0, 1), xycoords=('data', 'axes fraction'),
                     fontsize=8, ha='center', va='bottom')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = fig_dir / 'fig_shot_locked_average.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure 4: Success vs Failure shot-locked comparison ──────────────────────

def fig_success_vs_failure(session, fig_dir):
    """Side-by-side shot-locked averages for made vs missed, per band."""
    made = [s for s in session['shots'] if s['success']]
    missed = [s for s in session['shots'] if not s['success']]

    common_t = np.arange(-12, 8, 0.25)

    fig, axes = plt.subplots(len(BANDS), 1, figsize=(12, 12), sharex=True)
    fig.suptitle('Shot-Locked Power: Made vs Missed', fontsize=14, y=0.98)

    for ax, band in zip(axes, BANDS):
        for group, label, color, ls in [
            (made, 'Made', '#2ca02c', '-'),
            (missed, 'Missed', '#d62728', '-'),
        ]:
            all_interp = []
            for shot in group:
                t_epoch, p_epoch = _extract_shot_epoch(shot, band)
                if t_epoch is None:
                    continue
                interp_p = np.interp(common_t, t_epoch, p_epoch,
                                     left=np.nan, right=np.nan)
                all_interp.append(interp_p)
            if not all_interp:
                continue
            mat = np.array(all_interp)
            mean_p = np.nanmean(mat, axis=0)
            sem_p = np.nanstd(mat, axis=0) / np.sqrt(
                np.sum(~np.isnan(mat), axis=0).clip(min=1))

            ax.plot(common_t, mean_p, color=color, linewidth=1.5, linestyle=ls,
                    label=f'{label} (n={len(group)})')
            ax.fill_between(common_t, mean_p - sem_p, mean_p + sem_p,
                            color=color, alpha=0.15)

        ax.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.7)
        ax.set_ylabel(f'{band.capitalize()}\n(μV)', fontsize=9)
        ax.tick_params(labelsize=8)
        if band == 'delta':
            ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('Time relative to shot execution (s)', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = fig_dir / 'fig_success_vs_failure.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure 5: Phase-wise bar comparison (success vs failure) ─────────────────

def _mean_phase_power(shots, band, phase):
    vals = []
    for s in shots:
        entries = s['eegData'].get(phase, {}).get(band, [])
        if entries:
            vals.append(np.mean([e['power'] for e in entries]))
    return np.array(vals)


def fig_phase_bars(session, fig_dir):
    """Bar chart: mean power per phase, success vs failure, for each band."""
    made = [s for s in session['shots'] if s['success']]
    missed = [s for s in session['shots'] if not s['success']]
    phases_to_plot = ['preShot', 'recording', 'postShot']

    fig, axes = plt.subplots(1, len(BANDS), figsize=(16, 5), sharey=False)
    fig.suptitle('Mean Band Power by Phase: Made vs Missed', fontsize=13, y=1.02)

    x = np.arange(len(phases_to_plot))
    width = 0.35

    for ax, band in zip(axes, BANDS):
        made_means, made_sems = [], []
        missed_means, missed_sems = [], []

        for phase in phases_to_plot:
            m_vals = _mean_phase_power(made, band, phase)
            mi_vals = _mean_phase_power(missed, band, phase)
            made_means.append(np.mean(m_vals) if len(m_vals) else 0)
            made_sems.append(np.std(m_vals) / np.sqrt(len(m_vals)) if len(m_vals) > 1 else 0)
            missed_means.append(np.mean(mi_vals) if len(mi_vals) else 0)
            missed_sems.append(np.std(mi_vals) / np.sqrt(len(mi_vals)) if len(mi_vals) > 1 else 0)

        ax.bar(x - width / 2, made_means, width, yerr=made_sems,
               label='Made', color='#2ca02c', alpha=0.8, capsize=3)
        ax.bar(x + width / 2, missed_means, width, yerr=missed_sems,
               label='Missed', color='#d62728', alpha=0.8, capsize=3)
        ax.set_title(band.capitalize(), fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([PHASE_LABELS[p] for p in phases_to_plot], fontsize=7, rotation=25)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel('Mean Power (μV)', fontsize=9)
    axes[-1].legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    path = fig_dir / 'fig_phase_bars.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure 6: Theta/Alpha ratio ─────────────────────────────────────────────

def fig_theta_alpha_ratio(session, fig_dir):
    """Theta/alpha ratio over the session and per shot outcome."""
    ts = np.array(session['eegData']['timestamps'])
    theta_power = np.array(session['eegData']['bands']['theta'])
    alpha_power = np.array(session['eegData']['bands']['alpha'])
    ratio = theta_power / np.clip(alpha_power, 1e-6, None)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Theta / Alpha Ratio Analysis', fontsize=13, y=0.98)

    axes[0].plot(ts, ratio, color='#8c564b', linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel('θ/α Ratio', fontsize=10)
    axes[0].set_xlabel('Time (s)', fontsize=10)
    axes[0].set_title('Continuous θ/α Ratio Over Session', fontsize=11)
    for shot in session['shots']:
        color = '#2ca02c' if shot['success'] else '#d62728'
        axes[0].axvline(shot['duration'], color=color, linewidth=1, alpha=0.6, linestyle='--')

    made_ratios, missed_ratios = [], []
    for shot in session['shots']:
        pre_theta = np.mean([e['power'] for e in shot['eegData'].get('preShot', {}).get('theta', [])])
        pre_alpha = np.mean([e['power'] for e in shot['eegData'].get('preShot', {}).get('alpha', [])])
        r = pre_theta / max(pre_alpha, 1e-6)
        if shot['success']:
            made_ratios.append(r)
        else:
            missed_ratios.append(r)

    positions = [1, 2]
    bp = axes[1].boxplot([made_ratios, missed_ratios], positions=positions,
                         widths=0.5, patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ca02c')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#d62728')
    bp['boxes'][1].set_alpha(0.6)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f'Made (n={len(made_ratios)})', f'Missed (n={len(missed_ratios)})'])
    axes[1].set_ylabel('Pre-Shot θ/α Ratio', fontsize=10)
    axes[1].set_title('Pre-Shot θ/α Ratio by Outcome', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = fig_dir / 'fig_theta_alpha_ratio.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure 7: Learning curve / shot-by-shot progression ─────────────────────

def fig_shot_progression(session, fig_dir):
    """Band power trajectories across shots, coloured by outcome."""
    shots = session['shots']
    shot_nums = [s['shotNumber'] for s in shots]
    successes = [s['success'] for s in shots]

    fig, axes = plt.subplots(len(BANDS), 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Pre-Shot Band Power Across Shots', fontsize=13, y=0.98)

    for ax, band in zip(axes, BANDS):
        means = []
        for s in shots:
            vals = [e['power'] for e in s['eegData'].get('preShot', {}).get(band, [])]
            means.append(np.mean(vals) if vals else np.nan)

        ax.plot(shot_nums, means, 'o-', color=BAND_COLORS[band], linewidth=1.2, markersize=6)

        for i, (sn, m, suc) in enumerate(zip(shot_nums, means, successes)):
            marker_color = '#2ca02c' if suc else '#d62728'
            ax.plot(sn, m, 'o', color=marker_color, markersize=8, zorder=5)

        ax.set_ylabel(f'{band.capitalize()}\n(μV)', fontsize=9)
        ax.tick_params(labelsize=8)

        valid = [(sn, m) for sn, m in zip(shot_nums, means) if not np.isnan(m)]
        if len(valid) > 2:
            x_v, y_v = zip(*valid)
            z = np.polyfit(x_v, y_v, 1)
            ax.plot(shot_nums, np.polyval(z, shot_nums), '--', color='gray', alpha=0.5, linewidth=1)

    axes[-1].set_xlabel('Shot Number', fontsize=10)
    axes[-1].set_xticks(shot_nums)

    from matplotlib.lines import Line2D
    legend_el = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=8, label='Made'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=8, label='Missed'),
    ]
    axes[0].legend(handles=legend_el, fontsize=8, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = fig_dir / 'fig_shot_progression.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Figure 8: Shot outcome raster with phase shading ────────────────────────

def fig_shot_raster(session, fig_dir):
    """Visual overview: session timeline with shaded shot phases and outcomes."""
    ts = np.array(session['eegData']['timestamps'])
    alpha_power = np.array(session['eegData']['bands']['alpha'])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ts, alpha_power, color='#2ca02c', linewidth=0.4, alpha=0.5, label='Alpha power')

    phase_colors = {
        'prep': '#ffffcc', 'preShot': '#c7e9b4', 'recording': '#41b6c4',
        'postShot': '#fed976', 'review': '#e0e0e0'
    }

    for shot in session['shots']:
        for phase_name in PHASE_ORDER:
            phase = shot['eegData'].get(phase_name)
            if not phase:
                continue
            any_band = phase.get('delta', [])
            if not any_band:
                continue
            t_start = any_band[0]['timestamp']
            t_end = any_band[-1]['timestamp']
            rect = Rectangle((t_start, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0),
                              t_end - t_start, 200,
                              facecolor=phase_colors.get(phase_name, '#cccccc'),
                              alpha=0.3, edgecolor='none')
            ax.add_patch(rect)

        t_shot = shot['duration']
        color = '#2ca02c' if shot['success'] else '#d62728'
        symbol = '✓' if shot['success'] else '✗'
        ax.axvline(t_shot, color=color, linewidth=1.5, alpha=0.7)
        ax.text(t_shot, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] != 0 else 50,
                f'S{shot["shotNumber"]}{symbol}',
                fontsize=7, ha='center', color=color, fontweight='bold')

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Alpha Power (μV)', fontsize=10)
    ax.set_title('Session Timeline with Shot Phases', fontsize=12)
    ax.set_xlim(ts[0], ts[-1])
    plt.tight_layout()
    path = fig_dir / 'fig_shot_raster.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Statistical analysis ─────────────────────────────────────────────────────

def compute_statistics(session):
    """Compute descriptive and inferential stats for the report."""
    made = [s for s in session['shots'] if s['success']]
    missed = [s for s in session['shots'] if not s['success']]
    stats = {
        'n_total': session['totalShots'],
        'n_made': len(made),
        'n_missed': len(missed),
        'shooting_pct': len(made) / session['totalShots'] * 100,
        'session_duration_min': session['sessionDuration'] / 60,
        'player': session['playerName'],
        'band_stats': {},
    }

    for band in BANDS:
        band_s = {}
        made_pre = _mean_phase_power(made, band, 'preShot')
        missed_pre = _mean_phase_power(missed, band, 'preShot')

        band_s['made_preshot_mean'] = float(np.mean(made_pre)) if len(made_pre) else np.nan
        band_s['made_preshot_sd'] = float(np.std(made_pre, ddof=1)) if len(made_pre) > 1 else np.nan
        band_s['missed_preshot_mean'] = float(np.mean(missed_pre)) if len(missed_pre) else np.nan
        band_s['missed_preshot_sd'] = float(np.std(missed_pre, ddof=1)) if len(missed_pre) > 1 else np.nan

        if len(made_pre) >= 2 and len(missed_pre) >= 2:
            t_stat, p_val = ttest_ind(made_pre, missed_pre, equal_var=False)
            try:
                u_stat, u_p = mannwhitneyu(made_pre, missed_pre, alternative='two-sided')
            except ValueError:
                u_stat, u_p = np.nan, np.nan
            pooled_sd = np.sqrt(
                ((len(made_pre) - 1) * np.var(made_pre, ddof=1) +
                 (len(missed_pre) - 1) * np.var(missed_pre, ddof=1)) /
                (len(made_pre) + len(missed_pre) - 2)
            )
            cohens_d = (np.mean(made_pre) - np.mean(missed_pre)) / pooled_sd if pooled_sd > 0 else np.nan
        else:
            t_stat, p_val, u_stat, u_p, cohens_d = np.nan, np.nan, np.nan, np.nan, np.nan

        band_s['t_stat'] = float(t_stat)
        band_s['p_val'] = float(p_val)
        band_s['u_stat'] = float(u_stat)
        band_s['u_p'] = float(u_p)
        band_s['cohens_d'] = float(cohens_d)

        # Execution phase
        made_exec = _mean_phase_power(made, band, 'recording')
        missed_exec = _mean_phase_power(missed, band, 'recording')
        band_s['made_exec_mean'] = float(np.mean(made_exec)) if len(made_exec) else np.nan
        band_s['missed_exec_mean'] = float(np.mean(missed_exec)) if len(missed_exec) else np.nan

        stats['band_stats'][band] = band_s

    # Theta/alpha ratio
    made_ratios, missed_ratios = [], []
    for shot in session['shots']:
        pre_theta = [e['power'] for e in shot['eegData'].get('preShot', {}).get('theta', [])]
        pre_alpha = [e['power'] for e in shot['eegData'].get('preShot', {}).get('alpha', [])]
        if pre_theta and pre_alpha:
            r = np.mean(pre_theta) / max(np.mean(pre_alpha), 1e-6)
            if shot['success']:
                made_ratios.append(r)
            else:
                missed_ratios.append(r)

    stats['theta_alpha'] = {
        'made_mean': float(np.mean(made_ratios)) if made_ratios else np.nan,
        'made_sd': float(np.std(made_ratios, ddof=1)) if len(made_ratios) > 1 else np.nan,
        'missed_mean': float(np.mean(missed_ratios)) if missed_ratios else np.nan,
        'missed_sd': float(np.std(missed_ratios, ddof=1)) if len(missed_ratios) > 1 else np.nan,
    }

    # Shot progression correlation (pre-shot alpha vs shot number)
    shot_nums = []
    alpha_means = []
    for s in session['shots']:
        vals = [e['power'] for e in s['eegData'].get('preShot', {}).get('alpha', [])]
        if vals:
            shot_nums.append(s['shotNumber'])
            alpha_means.append(np.mean(vals))
    if len(shot_nums) > 2:
        r_corr, p_corr = pearsonr(shot_nums, alpha_means)
        stats['alpha_progression'] = {'r': float(r_corr), 'p': float(p_corr)}
    else:
        stats['alpha_progression'] = {'r': np.nan, 'p': np.nan}

    return stats


# ── Main runner ──────────────────────────────────────────────────────────────

def run_analysis(data_path, output_dir=None):
    data_path = Path(data_path)
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading session data from {data_path}...")
    session = load_session(data_path)
    print(f"  Player: {session['playerName']}")
    print(f"  Shots: {session['totalShots']}  "
          f"({sum(1 for s in session['shots'] if s['success'])} made, "
          f"{sum(1 for s in session['shots'] if not s['success'])} missed)")
    print(f"  Duration: {session['sessionDuration']:.1f}s")

    print("\nGenerating figures...")
    figures = {}
    figures['continuous_power'] = fig_continuous_power(session, fig_dir)
    print("  [1/8] Continuous band power")
    figures['raw_filtering'] = fig_raw_filtering(session, fig_dir)
    print("  [2/8] Raw signal + filtering")
    figures['shot_locked'] = fig_shot_locked_average(session, fig_dir)
    print("  [3/8] Shot-locked averages")
    figures['success_failure'] = fig_success_vs_failure(session, fig_dir)
    print("  [4/8] Success vs failure")
    figures['phase_bars'] = fig_phase_bars(session, fig_dir)
    print("  [5/8] Phase-wise bar chart")
    figures['theta_alpha'] = fig_theta_alpha_ratio(session, fig_dir)
    print("  [6/8] Theta/alpha ratio")
    figures['progression'] = fig_shot_progression(session, fig_dir)
    print("  [7/8] Shot progression")
    figures['raster'] = fig_shot_raster(session, fig_dir)
    print("  [8/8] Session raster")

    print("\nComputing statistics...")
    stats = compute_statistics(session)

    stats_path = output_dir / 'session_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Stats saved to {stats_path}")

    return figures, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze FreethrowEEG session')
    parser.add_argument('data_file', help='Path to session JSON file')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    args = parser.parse_args()
    run_analysis(args.data_file, args.output)
