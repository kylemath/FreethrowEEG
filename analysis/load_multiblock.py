"""
FreethrowEEG Multi-Block Session Loader
Loads multiple block JSON files from a session directory, validates data quality,
applies shot exclusions, and produces a unified session dict compatible with
the existing analysis pipeline.
"""

import json
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_RANGES = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 50)
}
PHASE_ORDER = ['prep', 'preShot', 'recording', 'postShot', 'review']

SESSION_DIR = Path(__file__).parent.parent / 'data' / 'session_Lukas_20250402_165214' / 'session2_lukas_2026'

BLOCK_FILES = [
    SESSION_DIR / 'freethrow_Lukas_Block1_2026-03-14.json',
    SESSION_DIR / 'freethrow_Lukas_Block2_2026-03-14.json',
    SESSION_DIR / 'freethrow_Lukas_Block3_2026-03-14.json',
]

VIDEO_FILES = {
    1: SESSION_DIR / 'freethrow_Lukas_Block1_2026-03-14.webm',
    3: SESSION_DIR / 'freethrow_Lukas_Block3_2026-03-14.webm',
}

EXCLUSIONS = {
    1: [18, 35],
    2: [34],
    3: [8, 21, 22, 25],
}


def _is_valid_shot(shot, max_sample_idx):
    """A shot is valid if its phase timestamps aren't all stuck at the buffer limit."""
    phases = shot.get('phases', {})
    if not phases:
        return True
    sample_indices = [phases[p]['sampleIdx'] for p in PHASE_ORDER if p in phases]
    if all(idx >= max_sample_idx for idx in sample_indices):
        return False
    return True


def _convert_eeg_entries(entries):
    """Convert {timestamp, value} dicts to {timestamp, power} dicts."""
    return [{'timestamp': e['timestamp'], 'power': e['value']} for e in entries]


def _convert_shot_eegdata(eeg_data):
    """Convert all phases/bands in a shot's eegData from value→power keys."""
    converted = {}
    for phase in PHASE_ORDER:
        if phase not in eeg_data:
            continue
        converted[phase] = {}
        for band in BANDS:
            entries = eeg_data[phase].get(band, [])
            if entries and isinstance(entries[0], dict) and 'value' in entries[0]:
                converted[phase][band] = _convert_eeg_entries(entries)
            elif entries and isinstance(entries[0], dict) and 'power' in entries[0]:
                converted[phase][band] = entries
            else:
                converted[phase][band] = entries
    return converted


def _compute_band_power_from_raw(raw_channels, sample_rate, window_sec=1.0, step_sec=0.25):
    """Compute continuous band power from raw electrode channels.

    Averages across electrodes, applies bandpass filters, and computes
    power in sliding windows.

    Returns (timestamps, bands_dict) where bands_dict maps band name to power array.
    """
    electrode_names = list(raw_channels.keys())
    if not electrode_names:
        return np.array([]), {b: np.array([]) for b in BANDS}

    min_len = min(len(raw_channels[e]) for e in electrode_names)
    signals = np.array([raw_channels[e][:min_len] for e in electrode_names], dtype=np.float64)
    avg_signal = np.mean(signals, axis=0)
    n_samples = len(avg_signal)

    if n_samples < sample_rate * 2:
        return np.array([]), {b: np.array([]) for b in BANDS}

    nyq = 0.5 * sample_rate
    window_samples = int(window_sec * sample_rate)
    step_samples = int(step_sec * sample_rate)

    n_windows = max(1, (n_samples - window_samples) // step_samples + 1)
    timestamps = np.array([(i * step_samples + window_samples / 2) / sample_rate
                           for i in range(n_windows)])

    bands_power = {}
    for band_name, (low, high) in BAND_RANGES.items():
        low_norm = low / nyq
        high_norm = high / nyq
        low_norm = max(low_norm, 0.001)
        high_norm = min(high_norm, 0.999)

        try:
            b, a = butter(4, [low_norm, high_norm], btype='bandpass')
            filtered = filtfilt(b, a, avg_signal)
        except Exception:
            bands_power[band_name] = np.zeros(n_windows)
            continue

        power = filtered ** 2

        band_windowed = np.zeros(n_windows)
        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            if end <= n_samples:
                band_windowed[i] = np.mean(power[start:end])
            else:
                band_windowed[i] = np.mean(power[start:n_samples])

        bands_power[band_name] = band_windowed

    return timestamps, bands_power


def _compute_raw_bands(raw_channels, sample_rate):
    """Compute raw (filtered, not squared) band signals for filtering demo.

    Returns dict mapping band name to filtered signal array.
    """
    electrode_names = list(raw_channels.keys())
    if not electrode_names:
        return {b: np.array([]) for b in BANDS}

    min_len = min(len(raw_channels[e]) for e in electrode_names)
    signals = np.array([raw_channels[e][:min_len] for e in electrode_names], dtype=np.float64)
    avg_signal = np.mean(signals, axis=0)

    nyq = 0.5 * sample_rate
    raw_bands = {}

    for band_name, (low, high) in BAND_RANGES.items():
        low_norm = max(low / nyq, 0.001)
        high_norm = min(high / nyq, 0.999)
        try:
            b, a = butter(4, [low_norm, high_norm], btype='bandpass')
            raw_bands[band_name] = filtfilt(b, a, avg_signal)
        except Exception:
            raw_bands[band_name] = np.zeros(len(avg_signal))

    return raw_bands


def load_block(filepath):
    """Load a single block JSON file."""
    with open(filepath) as f:
        return json.load(f)


def load_session_multiblock(block_files=None, exclusions=None):
    """Load and merge multiple block JSON files into a unified session.

    Returns:
        session: dict compatible with analyze_session.py format
        block_info: dict with block-level metadata (for video analysis)
    """
    if block_files is None:
        block_files = BLOCK_FILES
    if exclusions is None:
        exclusions = EXCLUSIONS

    block_files = [Path(f) for f in block_files]

    blocks = []
    for bf in block_files:
        if bf.exists():
            blocks.append(load_block(bf))
        else:
            print(f"  WARNING: Block file not found: {bf}")

    if not blocks:
        raise FileNotFoundError("No block files found")

    all_shots = []
    block_info = {}
    continuous_timestamps = []
    continuous_bands = {b: [] for b in BANDS}
    continuous_raw_bands = {b: [] for b in BANDS}
    continuous_raw_timestamps = []
    total_time_offset = 0.0
    total_raw_offset = 0.0
    global_shot_number = 0

    for block_idx, block_data in enumerate(blocks):
        block_num = block_idx + 1
        sample_rate = block_data.get('sampleRate', 256)
        total_samples = block_data.get('totalSamples', 0)
        max_sample_idx = total_samples

        block_exclusions = set(exclusions.get(block_num, []))

        ts_offset, band_power = _compute_band_power_from_raw(
            block_data.get('rawChannels', {}), sample_rate
        )
        raw_bands_block = _compute_raw_bands(
            block_data.get('rawChannels', {}), sample_rate
        )
        raw_timestamps_block = np.arange(len(list(raw_bands_block.values())[0])) / sample_rate if raw_bands_block else np.array([])

        if len(ts_offset) > 0:
            continuous_timestamps.append(ts_offset + total_time_offset)
            for b in BANDS:
                continuous_bands[b].append(band_power.get(b, np.zeros(len(ts_offset))))
        if len(raw_timestamps_block) > 0:
            continuous_raw_timestamps.append(raw_timestamps_block + total_raw_offset)
            for b in BANDS:
                rb = raw_bands_block.get(b, np.array([]))
                if len(rb) > 0:
                    continuous_raw_bands[b].append(rb)

        block_duration = block_data['timing']['sampleDurationSec']

        valid_count = 0
        excluded_count = 0
        invalid_count = 0
        block_shots_info = []

        for shot in block_data['shots']:
            shot_num_in_block = shot['shotNumber']

            if not _is_valid_shot(shot, max_sample_idx):
                invalid_count += 1
                continue

            if shot_num_in_block in block_exclusions:
                excluded_count += 1
                continue

            global_shot_number += 1
            valid_count += 1

            converted_eeg = _convert_shot_eegdata(shot['eegData'])

            rec_entries = converted_eeg.get('recording', {}).get('delta', [])
            if rec_entries:
                shot_time = rec_entries[0]['timestamp'] + total_time_offset
            else:
                shot_time = total_time_offset

            for phase in converted_eeg:
                for band in converted_eeg[phase]:
                    for entry in converted_eeg[phase][band]:
                        entry['timestamp'] += total_time_offset

            unified_shot = {
                'shotNumber': global_shot_number,
                'success': shot['success'],
                'duration': shot_time,
                'block': block_num,
                'originalShotNumber': shot_num_in_block,
                'eegData': converted_eeg,
            }
            all_shots.append(unified_shot)
            block_shots_info.append({
                'global_shot_number': global_shot_number,
                'block_shot_number': shot_num_in_block,
                'success': shot['success'],
                'recording_start_local': rec_entries[0]['timestamp'] - total_time_offset if rec_entries else None,
            })

        block_info[block_num] = {
            'file': str(block_files[block_idx]),
            'player_name': block_data.get('playerName', ''),
            'total_shots_in_block': block_data.get('totalShots', 0),
            'valid_shots': valid_count,
            'excluded_shots': excluded_count,
            'invalid_shots': invalid_count,
            'sample_rate': sample_rate,
            'sample_duration_sec': block_duration,
            'wall_clock_duration_sec': block_data['timing']['wallClockDurationSec'],
            'time_offset': total_time_offset,
            'raw_time_offset': total_raw_offset,
            'has_video': block_num in VIDEO_FILES and VIDEO_FILES[block_num].exists(),
            'video_path': str(VIDEO_FILES.get(block_num, '')),
            'shots': block_shots_info,
        }

        print(f"  Block {block_num}: {valid_count} valid, "
              f"{excluded_count} excluded, {invalid_count} invalid "
              f"(of {block_data.get('totalShots', 0)} total)")

        total_time_offset += block_duration + 10.0
        total_raw_offset += len(raw_timestamps_block) / sample_rate + 10.0 if len(raw_timestamps_block) > 0 else block_duration + 10.0

    if continuous_timestamps:
        merged_ts = np.concatenate(continuous_timestamps)
    else:
        merged_ts = np.array([])

    merged_bands = {}
    for b in BANDS:
        if continuous_bands[b]:
            merged_bands[b] = np.concatenate(continuous_bands[b]).tolist()
        else:
            merged_bands[b] = []

    merged_raw_bands = {}
    merged_raw_ts = np.array([])
    for b in BANDS:
        if continuous_raw_bands[b]:
            merged_raw_bands[b] = np.concatenate(continuous_raw_bands[b]).tolist()
        else:
            merged_raw_bands[b] = []
    if continuous_raw_timestamps:
        merged_raw_ts = np.concatenate(continuous_raw_timestamps)

    n_made = sum(1 for s in all_shots if s['success'])
    n_missed = sum(1 for s in all_shots if not s['success'])
    total_duration = sum(
        bi['sample_duration_sec'] for bi in block_info.values()
    )

    session = {
        'playerName': 'Lukas',
        'totalShots': len(all_shots),
        'sessionDuration': total_duration,
        'sampleRate': blocks[0].get('sampleRate', 256) if blocks else 256,
        'eegData': {
            'timestamps': merged_ts.tolist(),
            'bands': merged_bands,
            'rawBands': merged_raw_bands,
            'rawTimestamps': merged_raw_ts.tolist(),
        },
        'shots': all_shots,
        'blocks': block_info,
        'n_blocks': len(blocks),
        'n_made': n_made,
        'n_missed': n_missed,
    }

    return session, block_info


def get_video_shot_times(block_info):
    """Get shot times mapped to their respective video files for video analysis.

    Returns list of dicts with video_path and shot_times for each block with video.
    """
    video_blocks = []
    for block_num, info in block_info.items():
        if not info.get('has_video'):
            continue
        video_path = info['video_path']
        shots = info['shots']
        video_blocks.append({
            'block_num': block_num,
            'video_path': video_path,
            'shots': shots,
        })
    return video_blocks


if __name__ == '__main__':
    print("Loading multi-block session data...")
    session, block_info = load_session_multiblock()
    print(f"\nUnified session:")
    print(f"  Player: {session['playerName']}")
    print(f"  Total shots: {session['totalShots']} "
          f"({session['n_made']} made, {session['n_missed']} missed)")
    print(f"  Shooting %: {session['n_made']/session['totalShots']*100:.1f}%")
    print(f"  Session duration: {session['sessionDuration']:.1f}s "
          f"({session['sessionDuration']/60:.1f} min)")
    print(f"  Continuous timestamps: {len(session['eegData']['timestamps'])} points")
    print(f"  Blocks: {session['n_blocks']}")
    for bnum, bi in block_info.items():
        print(f"    Block {bnum}: {bi['valid_shots']} shots, "
              f"video={'yes' if bi['has_video'] else 'no'}")

    print(f"\nShot breakdown:")
    for shot in session['shots']:
        label = 'made' if shot['success'] else 'missed'
        print(f"  Shot {shot['shotNumber']:3d} (B{shot['block']}.{shot['originalShotNumber']:2d}) "
              f"{label:6s} t={shot['duration']:.1f}s")
