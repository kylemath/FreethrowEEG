"""
FreethrowEEG Report Generator
Runs the analysis, populates the LaTeX template with results, and compiles to PDF.
"""

import json
import subprocess
import shutil
import sys
from datetime import datetime
from pathlib import Path
import math
import argparse

SCRIPT_DIR = Path(__file__).parent


def _fmt(val, decimals=2):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return '---'
    return f'{val:.{decimals}f}'


def _p_fmt(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return '---'
    if p < 0.001:
        return '$<$.001'
    return f'{p:.3f}'


def _interpret_direction(made_mean, missed_mean, band):
    if math.isnan(made_mean) or math.isnan(missed_mean):
        return "insufficient data to compare"
    diff_pct = (made_mean - missed_mean) / missed_mean * 100
    direction = "higher" if diff_pct > 0 else "lower"
    return (f"{band} power was {abs(diff_pct):.1f}\\% {direction} for made shots "
            f"($M = {made_mean:.1f}$) compared to missed ($M = {missed_mean:.1f}$)")


def build_stats_table(stats):
    rows = []
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        bs = stats['band_stats'][band]
        made_str = f"{_fmt(bs['made_preshot_mean'],1)} ({_fmt(bs['made_preshot_sd'],1)})"
        missed_str = f"{_fmt(bs['missed_preshot_mean'],1)} ({_fmt(bs['missed_preshot_sd'],1)})"
        rows.append(
            f"{band.capitalize()} & {made_str} & {missed_str} & "
            f"{_fmt(bs['t_stat'])} & {_p_fmt(bs['p_val'])} & "
            f"{_fmt(bs['u_stat'],0)} & {_fmt(bs['cohens_d'])} \\\\"
        )
    return '\n'.join(rows)


def build_continuous_interp(stats):
    lines = []
    lines.append(
        "Visual inspection of the continuous traces reveals characteristic "
        "fluctuations across all five bands over the "
        f"{_fmt(stats['session_duration_min'],1)}-minute session."
    )
    return ' '.join(lines)


def build_made_vs_missed_interp(stats):
    parts = []
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        bs = stats['band_stats'][band]
        parts.append(_interpret_direction(bs['made_preshot_mean'], bs['missed_preshot_mean'], band))

    sig_bands = [b for b in ['delta', 'theta', 'alpha', 'beta', 'gamma']
                 if stats['band_stats'][b]['p_val'] < 0.1 and not math.isnan(stats['band_stats'][b]['p_val'])]
    if sig_bands:
        parts.append(
            f"Trend-level differences ($p < .10$) were observed for: "
            f"{', '.join(b.capitalize() for b in sig_bands)}"
        )
    else:
        parts.append(
            "No frequency band reached trend-level significance ($p < .10$) in the "
            "pre-shot comparison, consistent with the limited sample size"
        )

    return '. '.join(parts) + '.'


def build_phase_interp(stats):
    parts = []
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        bs = stats['band_stats'][band]
        m_pre = bs['made_preshot_mean']
        m_exec = bs['made_exec_mean']
        if not (math.isnan(m_pre) or math.isnan(m_exec)):
            change_pct = (m_exec - m_pre) / m_pre * 100
            direction = "increased" if change_pct > 0 else "decreased"
            parts.append(
                f"For made shots, {band} power {direction} by "
                f"{abs(change_pct):.1f}\\% from pre-shot to execution"
            )
    return '. '.join(parts) + '.' if parts else ''


def build_theta_alpha_interp(stats):
    ta = stats['theta_alpha']
    parts = [
        "The pre-shot theta/alpha ratio has been proposed as an index of "
        "focused attention in sport performance."
    ]
    if not math.isnan(ta['made_mean']):
        parts.append(
            f"Mean pre-shot $\\theta/\\alpha$ ratio was "
            f"{_fmt(ta['made_mean'])} (SD = {_fmt(ta['made_sd'])}) for made shots "
            f"and {_fmt(ta['missed_mean'])} (SD = {_fmt(ta['missed_sd'])}) for missed shots"
        )
        diff = ta['made_mean'] - ta['missed_mean']
        direction = "higher" if diff > 0 else "lower"
        parts.append(
            f"Made shots showed a {direction} $\\theta/\\alpha$ ratio, "
            f"differing by {abs(diff):.2f}"
        )
    return '. '.join(parts) + '.'


def build_progression_interp(stats):
    ap = stats['alpha_progression']
    parts = []
    if not math.isnan(ap['r']):
        direction = "increase" if ap['r'] > 0 else "decrease"
        parts.append(
            f"Pre-shot alpha power showed a trend toward {direction} across "
            f"the session ($r = {ap['r']:.2f}$, $p = {_p_fmt(ap['p'])}$)"
        )
        if abs(ap['r']) > 0.5:
            parts.append(
                "suggesting a moderate temporal drift that could reflect "
                "fatigue, habituation, or learning effects"
            )
        else:
            parts.append(
                "though the correlation was weak, possibly due to the "
                "limited number of trials"
            )
    return ', '.join(parts) + '.' if parts else ''


def build_kinematics_interp(stats):
    video = stats.get('video_analysis', {})
    comp = video.get('comparison', {})
    parts = []
    for feat, label in [('elbow_angle', 'elbow angle'), ('wrist_height', 'wrist height'),
                         ('knee_angle', 'knee angle'), ('body_lean_angle', 'body lean')]:
        c = comp.get(feat, {})
        m_peak = c.get('made_peak_mean')
        mi_peak = c.get('missed_peak_mean')
        if m_peak is not None and mi_peak is not None:
            direction = 'higher' if m_peak > mi_peak else 'lower'
            parts.append(
                f"Peak {label} was {direction} for made shots "
                f"({m_peak:.1f}) than missed ({mi_peak:.1f})"
            )
    if not parts:
        return "Pose kinematics showed descriptive differences between made and missed shots."
    return '. '.join(parts) + '.'


def build_poseeeg_interp(stats):
    return (
        "Time-aligned visualization of pose and EEG data during execution "
        "reveals concurrent motor and neural dynamics. Cross-domain correlations "
        "between mean pose features and alpha power are annotated where available."
    )


def build_discussion(stats):
    paragraphs = []

    paragraphs.append(
        f"This pilot session ({stats['n_total']} shots, {stats['shooting_pct']:.0f}\\% accuracy) "
        f"provides an initial demonstration that the FreethrowEEG system can capture "
        f"frequency-band-specific EEG dynamics during free-throw shooting."
    )

    notable = []
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        bs = stats['band_stats'][band]
        if not math.isnan(bs['cohens_d']) and abs(bs['cohens_d']) > 0.5:
            notable.append((band, bs['cohens_d']))
    if notable:
        band_strs = [f"{b} ($d = {d:.2f}$)" for b, d in notable]
        paragraphs.append(
            f"Medium-to-large effect sizes were observed for pre-shot power in "
            f"{', '.join(band_strs)}, suggesting these bands may carry "
            f"performance-relevant information even in a small sample."
        )
    else:
        paragraphs.append(
            "Effect sizes for pre-shot band power differences were generally "
            "small, consistent with the limited statistical power of a single "
            "10-shot session."
        )

    ta = stats['theta_alpha']
    if not math.isnan(ta['made_mean']) and not math.isnan(ta['missed_mean']):
        diff = ta['made_mean'] - ta['missed_mean']
        if abs(diff) > 0.05:
            direction = "higher" if diff > 0 else "lower"
            paragraphs.append(
                f"The $\\theta/\\alpha$ ratio was {direction} for made shots, "
                f"which is broadly consistent with prior literature linking "
                f"theta/alpha dynamics to attentional focus during precision tasks. "
                f"Replication with more trials is essential."
            )

    paragraphs.append(
        "The shot-by-shot progression analysis provides a preliminary look at "
        "temporal dynamics across the session. Changes in band power over "
        "successive shots may reflect fatigue, adaptation, or strategy "
        "adjustment, though the current data cannot distinguish between these."
    )

    video = stats.get('video_analysis', {})
    comp = video.get('comparison', {})
    if comp:
        paragraphs.append(
            "Video-based pose estimation using MediaPipe revealed descriptive "
            "differences in shooting kinematics between made and missed shots. "
            "Stop-motion decomposition and ghost-overlay composites provide "
            "visual evidence of form consistency, while time-aligned pose and "
            "EEG traces offer a preliminary window into the brain--body "
            "dynamics underlying free-throw execution."
        )

    return '\n\n'.join(paragraphs)


def build_conclusion_extra(stats):
    notable = [b for b in ['delta', 'theta', 'alpha', 'beta', 'gamma']
               if not math.isnan(stats['band_stats'][b]['cohens_d'])
               and abs(stats['band_stats'][b]['cohens_d']) > 0.3]
    if notable:
        return (
            f"Exploratory analyses identified {', '.join(b.capitalize() for b in notable)} "
            f"band power as potentially differentiating made from missed shots."
        )
    return (
        "While no strong differentiators were found in this single session, "
        "the data pipeline is validated for larger-scale collection."
    )


def populate_template(template_path, stats, output_tex_path):
    with open(template_path) as f:
        tex = f.read()

    replacements = {
        '%%DATE%%': datetime.now().strftime('%B %d, %Y'),
        '%%NTOTAL%%': str(stats['n_total']),
        '%%NMADE%%': str(stats['n_made']),
        '%%NMISSED%%': str(stats['n_missed']),
        '%%SHOOTPCT%%': _fmt(stats['shooting_pct'], 0),
        '%%PLAYER%%': stats['player'].replace('_', '\\_'),
        '%%DURATION%%': _fmt(stats['session_duration_min'], 1),
        '%%ABSTRACT_EXTRA%%': '',
        '%%STATS_TABLE_ROWS%%': build_stats_table(stats),
        '%%CONTINUOUS_INTERP%%': build_continuous_interp(stats),
        '%%MADEVSMISSED_INTERP%%': build_made_vs_missed_interp(stats),
        '%%PHASEBARS_INTERP%%': build_phase_interp(stats),
        '%%THETA_ALPHA_INTERP%%': build_theta_alpha_interp(stats),
        '%%PROGRESSION_INTERP%%': build_progression_interp(stats),
        '%%DISCUSSION%%': build_discussion(stats),
        '%%CONCLUSION_EXTRA%%': build_conclusion_extra(stats),
        '%%KINEMATICS_INTERP%%': build_kinematics_interp(stats),
        '%%POSEEEG_INTERP%%': build_poseeeg_interp(stats),
    }

    for placeholder, value in replacements.items():
        tex = tex.replace(placeholder, value)

    with open(output_tex_path, 'w') as f:
        f.write(tex)

    return output_tex_path


def compile_pdf(tex_path):
    """Compile .tex to PDF using pdflatex. Falls back to latexmk."""
    tex_dir = tex_path.parent
    tex_name = tex_path.stem

    for compiler in ['pdflatex', 'latexmk']:
        exe = shutil.which(compiler)
        if exe is None:
            continue

        if compiler == 'pdflatex':
            cmd = [exe, '-interaction=nonstopmode', '-halt-on-error', tex_path.name]
        else:
            cmd = [exe, '-pdf', '-interaction=nonstopmode', tex_path.name]

        print(f"  Compiling with {compiler}...")
        for pass_num in range(2):
            result = subprocess.run(cmd, cwd=tex_dir, capture_output=True, text=True, timeout=120)
            if result.returncode != 0 and pass_num == 1:
                print(f"  WARNING: {compiler} returned non-zero exit code.")
                print(f"  Check {tex_dir / (tex_name + '.log')} for details.")

        pdf_path = tex_dir / (tex_name + '.pdf')
        if pdf_path.exists():
            print(f"  PDF generated: {pdf_path}")
            return pdf_path
        else:
            print(f"  {compiler} did not produce a PDF, trying next compiler...")

    print("ERROR: No LaTeX compiler found. Install texlive or mactex.")
    print("  On macOS:  brew install --cask mactex-no-gui")
    print("  On Ubuntu: sudo apt install texlive-full")
    return None


def main():
    parser = argparse.ArgumentParser(description='Generate FreethrowEEG analysis report')
    parser.add_argument('data_file', help='Path to session JSON file')
    parser.add_argument('--output', '-o', default=None, help='Output directory (default: analysis/)')
    parser.add_argument('--skip-compile', action='store_true', help='Skip PDF compilation')
    parser.add_argument('--video-results', default=None,
                        help='Path to video_analysis_results.json from run_video_analysis.py')
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(SCRIPT_DIR))
    from analyze_session import run_analysis

    figures, stats = run_analysis(args.data_file, output_dir)

    video_results_path = args.video_results
    if video_results_path is None:
        default_path = SCRIPT_DIR / 'video_output' / 'video_analysis_results.json'
        if default_path.exists():
            video_results_path = str(default_path)

    if video_results_path:
        print(f"\nLoading video analysis results from {video_results_path}...")
        with open(video_results_path) as f:
            stats['video_analysis'] = json.load(f)
    else:
        stats['video_analysis'] = {}

    print("\nPopulating LaTeX template...")
    template_path = SCRIPT_DIR / 'report_template.tex'
    output_tex = output_dir / 'report.tex'
    populate_template(template_path, stats, output_tex)
    print(f"  LaTeX source: {output_tex}")

    if not args.skip_compile:
        print("\nCompiling PDF...")
        pdf = compile_pdf(output_tex)
        if pdf:
            print(f"\n  Report ready: {pdf}")
        else:
            print(f"\n  LaTeX source is ready at {output_tex}")
            print("  Compile manually: cd analysis && pdflatex report.tex")
    else:
        print(f"\n  Skipped compilation. LaTeX source: {output_tex}")
        print(f"  Compile manually: cd {output_dir} && pdflatex report.tex")


if __name__ == '__main__':
    main()
