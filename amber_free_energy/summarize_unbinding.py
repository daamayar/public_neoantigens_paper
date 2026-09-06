#!/usr/bin/env python3
"""
Summarize peptide-terminal unbinding detection results across all complexes.

Reads the per-complex unbinding summary JSON files from the ``results/`` tree
(produced by ``scripts/detect_unbinding.py``) and aggregates the 10-replica
statistics into a single cross-complex table. The script mirrors the CLI and
output conventions of ``summarize_mmpbsa.py``.

Per-complex columns
-------------------
  - n_replicas_total:           number of replicas analysed
  - n_replicas_with_event:      replicas where at least one terminal reached
                                the unbound state over a sustain window
  - fraction_unbound:           n_replicas_with_event / n_replicas_total
  - t_first_unbind_mean_ns:     mean of per-replica t_first_unbind (replicas
                                without an event are excluded from the mean)
  - t_first_unbind_sem_ns:      SEM of the same set
  - t_first_unbind_median_ns:   median of per-replica t_first_unbind (events
                                only)
  - mean_fraction_time_unbound: average over replicas of the per-replica
                                fraction_time_unbound (N-term OR C-term)
  - terminal_breakdown:         N-only / C-only / Both / Stable counts

Sign convention note: there is no sign convention to flag here — times are in
ns and fractions are dimensionless. Replicas without an event (``t = null``)
are counted in the breakdown (``Stable``) and excluded from the t-statistics.

Output formats
--------------
  - Terminal: colored table (ANSI)
  - CSV:      ``unbinding_summary_across_complexes_{criterion}_{scope}.csv``
  - Markdown: ``unbinding_summary_across_complexes_{criterion}_{scope}.md``

Usage
-----
    python summarize_unbinding.py


    # Results directory / output paths
    python summarize_unbinding.py --results-dir /path/to/results
    python summarize_unbinding.py --csv mysummary.csv --md mysummary.md

    # Summarize only specific complexes
    python summarize_unbinding.py --complexes flt3_d835y_neo_a_0201_yimsdsnyv flt3_d835y_wt_a_0201_dimsdsnyv

    # Typical command
"""

import argparse
import csv
import json
import os
import sys


# Default results directory. Taken from scripts.config so that this standalone
# summary always reads the directory the pipeline actually wrote to, including
# when it has been redirected with PHLA_RESULTS_DIR.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from scripts import config as _config          # noqa: E402
DEFAULT_RESULTS_DIR = _config.RESULTS_DIR

# Known neo/wt pairs: (neo_dir_name, wt_dir_name, display_label). Kept in sync
# with summarize_mmpbsa.py so the two summaries present the same row order.
PAIRS = [
    ("flt3_d835y_neo_a_0201_yimsdsnyv",    "flt3_d835y_wt_a_0201_dimsdsnyv",     "FLT3 D835Y – A*02:01 – YIMSDSNYV / DIMSDSNYV"),
    ("pik3ca_e545k_neo_a_1101_strdplseitk", "pik3ca_e545k_wt_a_1101_strdplseite", "PIK3CA E545K – A*11:01 – STRDPLSEITK / STRDPLSEITE"),
    ("kras_g12c_neo_a_1101_vvvgacgvgk",    "kras_g12c_wt_a_1101_vvvgaggvgk",     "KRAS G12C – A*11:01 – VVVGACGVGK / VVVGAGGVGK"),
    ("kras_g12d_neo_c_0802_gadgvgksa",     "kras_g12d_wt_c_0802_gaggvgksa",      "KRAS G12D – C*08:02 – GADGVGKSA / GAGGVGKSA"),
    ("kras_g12v_neo_a_1101_vvgavgvgk",     "kras_g12v_wt_a_1101_vvgaggvgk",      "KRAS G12V – A*11:01 – VVGAVGVGK / VVGAGGVGK"),
    ("p53_r175h_neo_a_0201_hmtevvrhc",     "p53_r175h_wt_a_0201_hmtevvrrc",      "p53 R175H – A*02:01 – HMTEVVRHC / HMTEVVRRC"),
    ("pik3ca_h1047l_neo_a_0301_alhggwttk", "pik3ca_h1047l_wt_a_0301_ahhggwttk",  "PIK3CA H1047L – A*03:01 – ALHGGWTTK / AHHGGWTTK"),
]

UNPAIRED = [
    ("apc_neo_a_0201_lqmdflvhpa",    "APC neo – A*02:01 – LQMDFLVHPA"),
    ("npm_neo_a_0201_claveevsl",     "NPM neo – A*02:01 – CLAVEEVSL"),
    ("tgfbrii_neo_a_0201_rlsscvpva", "TGFBRii neo – A*02:01 – RLSSCVPVA"),
]


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

# detect_unbinding.py fixes the criterion and the terminal scope, so these are
# constants here too. They name every stored output file.
CRITERION = "dist-contact"
SCOPE = "residue+neighbor"


def _filename_suffix(criterion, scope):
    """Return the ``{criterion}_{scope}`` suffix used by detect_unbinding.py.

    The scope may contain a '+' (``residue+neighbor``) which is translated to
    '-' for filesystem safety — this matches the suffix emitted by
    ``UnbindingParams.suffix`` in ``scripts/detect_unbinding.py``.
    """
    return f"{criterion}_{scope.replace('+', '-')}"


def load_complex_data(results_dir, complex_name, criterion, scope):
    """Load the unbinding summary JSON for a single complex.

    Args:
        results_dir (str): Top-level results directory (contains one
            subdirectory per complex).
        complex_name (str): Complex name (directory name under results_dir).

    Returns:
        dict or None: a dict with the per-complex aggregate fields, or None if
        the JSON file is missing or unreadable. Keys returned:
            - n_replicas_total, n_replicas_with_event, fraction_unbound
            - t_first_unbind_mean_ns, t_first_unbind_sem_ns,
              t_first_unbind_median_ns
            - mean_fraction_time_unbound
            - nterm_only, cterm_only, both, neither
            - criterion, scope (echoed from the loaded JSON)
    """
    suffix = _filename_suffix(criterion, scope)
    json_path = os.path.join(results_dir, complex_name,
                             f"unbinding_summary_{suffix}.json")
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {json_path}: {exc}", file=sys.stderr)
        return None

    agg = data.get("aggregate")
    if not agg:
        return None
    tb = agg.get("terminal_breakdown", {})

    # Record the params actually used (for the footer; detect what thresholds
    # were in effect when these results were produced).
    params = data.get("params", {})

    return {
        "n_replicas_total": agg.get("n_replicas_total", 0),
        "n_replicas_with_event": agg.get("n_replicas_with_event", 0),
        "fraction_unbound": agg.get("fraction_unbound", 0.0),
        "t_first_unbind_mean_ns": agg.get("t_first_unbind_mean_ns"),
        "t_first_unbind_sem_ns": agg.get("t_first_unbind_sem_ns"),
        "t_first_unbind_median_ns": agg.get("t_first_unbind_median_ns"),
        "mean_fraction_time_unbound": agg.get("mean_fraction_time_unbound", 0.0),
        "nterm_only": tb.get("nterm_only", 0),
        "cterm_only": tb.get("cterm_only", 0),
        "both": tb.get("both", 0),
        "neither": tb.get("neither", 0),
        "params": params,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_t(mean, sem):
    """Format a time (ns) mean/SEM pair, or 'stable' if there were no events."""
    if mean is None:
        return "  stable  "
    if sem is None:
        return f"{mean:>7.1f}        "
    return f"{mean:>7.1f} +/- {sem:>5.1f}"


def fmt_med(med):
    """Format a median time in ns, or '—' if unavailable."""
    if med is None:
        return "   —   "
    return f"{med:>7.1f}"


def fmt_frac(v):
    """Format a fraction as a percentage string."""
    if v is None:
        return "  —  "
    return f"{100.0 * v:>5.1f}%"


def fmt_breakdown(d):
    """Format the N/C/Both/Stable breakdown as a compact 'a/b/c/d' string.

    Order matches the header: N-only / C-only / Both / Stable.
    """
    if d is None:
        return "—"
    return f"{d['nterm_only']}/{d['cterm_only']}/{d['both']}/{d['neither']}"


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_terminal_table(rows, criterion, scope):
    """Print a colored summary table to stdout.

    Args:
        rows (list[dict]): Row dicts with keys: label, type ('neo' | 'wt' |
            'unpaired' | 'separator' | 'section_header'), data (loaded by
            ``load_complex_data`` or None).
        criterion (str): Criterion tag for the title banner.
        scope (str): Scope tag for the title banner.
    """
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    header = (f"{'Complex':<50s} "
              f"{'Reps':>4s}  "
              f"{'Events':>7s}  "
              f"{'t_first (mean ns)':>20s}  "
              f"{'median':>8s}  "
              f"{'frac_t_unb':>10s}  "
              f"{'N/C/Both/Stable':>16s}")
    sep_line = "-" * len(header)

    print()
    print(f"{BOLD}Unbinding Detection Summary — criterion={criterion}, "
          f"scope={scope}{RESET}")
    print(sep_line)
    print(f"{BOLD}{header}{RESET}")
    print(sep_line)

    for row in rows:
        rtype = row["type"]
        label = row["label"]
        d = row.get("data")

        if rtype == "separator":
            print(sep_line)
            continue
        if rtype == "section_header":
            print(f"\n{BOLD}{CYAN}{label}{RESET}")
            print(sep_line)
            continue

        if d is None:
            print(f"  {label:<48s} {'—':>4s}  {'NO DATA':>7s}")
            continue

        # Choose color: high fraction unbound → red, none → green
        frac = d.get("fraction_unbound") or 0.0
        if frac >= 0.5:
            color = RED
        elif frac == 0.0:
            color = GREEN
        else:
            color = YELLOW
        prefix = "  " if rtype in ("neo", "wt") else ""
        warn = f" {YELLOW}(!){RESET}" if d["n_replicas_total"] < 10 else ""

        print(f"{prefix}{label:<48s} "
              f"{d['n_replicas_total']:>4d}  "
              f"{color}{d['n_replicas_with_event']:>2d}/{d['n_replicas_total']:<2d} "
              f"{fmt_frac(d['fraction_unbound']):>5s}{RESET}  "
              f"{fmt_t(d['t_first_unbind_mean_ns'], d['t_first_unbind_sem_ns']):>20s}  "
              f"{fmt_med(d['t_first_unbind_median_ns']):>8s}  "
              f"{fmt_frac(d['mean_fraction_time_unbound']):>10s}  "
              f"{fmt_breakdown(d):>16s}"
              f"{warn}")

    print(sep_line)
    print(f"{DIM}Legend: Events = replicas_with_event / n_replicas_total "
          f"(percentage); t_first = mean time (ns) of first sustained "
          f"unbinding over replicas with events; N/C/Both/Stable = "
          f"terminal-level breakdown across replicas.{RESET}")
    print()


# ---------------------------------------------------------------------------
# CSV / Markdown output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "complex", "type",
    "n_replicas_total", "n_replicas_with_event", "fraction_unbound",
    "t_first_unbind_mean_ns", "t_first_unbind_sem_ns",
    "t_first_unbind_median_ns",
    "mean_fraction_time_unbound",
    "nterm_only", "cterm_only", "both", "neither",
]


def write_csv(rows, output_path):
    """Write a machine-readable CSV containing the per-complex aggregates.

    Args:
        rows (list[dict]): Same row list used by ``print_terminal_table``.
        output_path (str): Destination CSV file path.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for row in rows:
            if row["type"] in ("separator", "section_header"):
                continue
            d = row.get("data")
            if d is None:
                writer.writerow({"complex": row["label"], "type": row["type"]})
                continue
            csv_row = {"complex": row["label"], "type": row["type"]}
            for key in CSV_FIELDS[2:]:
                csv_row[key] = d.get(key, "")
            writer.writerow(csv_row)

    print(f"CSV written to {output_path}")


def _md_fmt_t(mean, sem):
    if mean is None:
        return "stable"
    if sem is None:
        return f"{mean:.1f}"
    return f"{mean:.1f} +/- {sem:.1f}"


def _md_fmt_frac(v):
    if v is None:
        return "—"
    return f"{100.0 * v:.1f}%"


def write_markdown(rows, criterion, scope, output_path):
    """Write a Markdown report summarising results across complexes.

    Args:
        rows (list[dict]): Row list used by ``print_terminal_table``.
        criterion (str): Criterion used for this summary.
        scope (str): Terminal scope used for this summary.
        output_path (str): Destination Markdown file path.
    """
    header = (
        "| Complex | Reps | Events | t_first (mean ns) | t_first median "
        "| frac_t_unbound | N-only / C-only / Both / Stable |"
    )
    divider = (
        "|---------|------|--------|-------------------|---------------"
        "|----------------|-------------------------------|"
    )

    with open(output_path, "w") as f:
        f.write(
            f"# Peptide-Terminal Unbinding Summary "
            f"(criterion={criterion}, scope={scope})\n\n"
        )
        f.write(
            "Times in ns; fractions as percentages; counts are per-replica "
            "over the 10-replica ensemble. Replicas without an event "
            "contribute to the *Stable* column and are excluded from the "
            "t_first statistics.\n\n"
        )

        f.write("## Per-Complex Results\n\n")
        f.write(header + "\n")
        f.write(divider + "\n")

        for row in rows:
            if row["type"] == "separator":
                continue
            if row["type"] == "section_header":
                f.write(f"\n**{row['label']}**\n\n")
                f.write(header + "\n")
                f.write(divider + "\n")
                continue

            d = row.get("data")
            label = row["label"]
            if d is None:
                f.write(f"| {label} | — | NO DATA | | | | |\n")
                continue

            events_cell = (
                f"{d['n_replicas_with_event']}/{d['n_replicas_total']} "
                f"({_md_fmt_frac(d['fraction_unbound'])})"
            )
            median_val = d["t_first_unbind_median_ns"]
            median_cell = "—" if median_val is None else f"{median_val:.1f}"
            t_cell = _md_fmt_t(d["t_first_unbind_mean_ns"],
                               d["t_first_unbind_sem_ns"])
            frac_cell = _md_fmt_frac(d["mean_fraction_time_unbound"])
            breakdown_cell = (
                f"{d['nterm_only']} / {d['cterm_only']} / "
                f"{d['both']} / {d['neither']}"
            )
            f.write(
                f"| {label} | {d['n_replicas_total']} | {events_cell} | "
                f"{t_cell} | {median_cell} | {frac_cell} | {breakdown_cell} |\n"
            )

        f.write("\n---\n")
        f.write("*Generated by `summarize_unbinding.py`.*\n")

    print(f"Markdown written to {output_path}")


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def build_rows(results_dir, criterion, scope, complex_filter=None):
    """Assemble the row list for display.

    Args:
        results_dir (str): Top-level results directory.
        criterion (str): Criterion tag ('dist-contact' | 'rmsd').
        scope (str): Scope tag ('residue' | 'residue+neighbor').
        complex_filter (set[str] | None): If given, only these directory names
            are included in the output.

    Returns:
        list[dict]: Rows with keys ``label``, ``type``, optional ``data``.
    """
    rows = []
    allow = complex_filter  # may be None or a set of allowed names

    def _allowed(name):
        return allow is None or name in allow

    # --- Paired complexes ---
    rows.append({"label": "Paired Complexes (neo vs wt)", "type": "section_header"})
    for neo_name, wt_name, pair_label in PAIRS:
        parts = pair_label.split(" – ")
        hla = parts[0] if len(parts) > 0 else ""
        peptides = parts[-1] if len(parts) > 1 else ""
        mutation = parts[1] if len(parts) > 2 else ""
        prefix = f"{mutation} – " if mutation else ""
        neo_pep = peptides.split(" / ")[0] if " / " in peptides else neo_name
        wt_pep = peptides.split(" / ")[1] if " / " in peptides else wt_name
        neo_label = f"{prefix}{hla} – {neo_pep} (neo)"
        wt_label = f"{prefix}{hla} – {wt_pep} (wt)"

        if _allowed(neo_name):
            neo_data = load_complex_data(results_dir, neo_name, criterion, scope)
            rows.append({"label": neo_label, "type": "neo", "data": neo_data})
        if _allowed(wt_name):
            wt_data = load_complex_data(results_dir, wt_name, criterion, scope)
            rows.append({"label": wt_label, "type": "wt", "data": wt_data})

        rows.append({"label": "", "type": "separator"})

    # --- Unpaired complexes ---
    rows.append({"label": "Unpaired Complexes", "type": "section_header"})
    for name, label in UNPAIRED:
        if not _allowed(name):
            continue
        data = load_complex_data(results_dir, name, criterion, scope)
        rows.append({"label": label, "type": "unpaired", "data": data})

    # --- Auto-detected extras ---
    known = set()
    for neo, wt, _ in PAIRS:
        known.add(neo)
        known.add(wt)
    for name, _ in UNPAIRED:
        known.add(name)

    extra = []
    if os.path.isdir(results_dir):
        for d in sorted(os.listdir(results_dir)):
            if d.endswith("_bkp"):
                continue
            if d in known:
                continue
            if not _allowed(d):
                continue
            data = load_complex_data(results_dir, d, criterion, scope)
            if data is not None:
                extra.append({"label": d, "type": "unpaired", "data": data})

    if extra:
        rows.append({"label": "", "type": "separator"})
        rows.append({"label": "Other Complexes (auto-detected)",
                     "type": "section_header"})
        rows.extend(extra)

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize peptide-terminal unbinding detection results across "
            "all complexes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default (dist-contact, residue scope)
    python summarize_unbinding.py

    # RMSD criterion

    # Residue+neighbor scope
    python summarize_unbinding.py --scope residue+neighbor

    # Write CSV and Markdown
    python summarize_unbinding.py --csv out.csv --md out.md

    # Summarize only one complex
    python summarize_unbinding.py --complexes flt3_d835y_neo_a_0201_yimsdsnyv
        """,
    )
    parser.add_argument(
        "--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--complexes", type=str, nargs="+", default=None,
        help="Limit summary to the listed complex directory names.",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help=(
            "Output CSV file path. If omitted and --auto-names is set, "
            "writes unbinding_summary_across_complexes_{criterion}_{scope}.csv"
            " next to this script."
        ),
    )
    parser.add_argument(
        "--md", type=str, default=None,
        help=(
            "Output Markdown file path. If omitted and --auto-names is set, "
            "writes unbinding_summary_across_complexes_{criterion}_{scope}.md"
            " next to this script."
        ),
    )
    parser.add_argument(
        "--auto-names", action="store_true",
        help=(
            "Auto-derive --csv and --md paths from --criterion/--scope when "
            "not explicitly provided."
        ),
    )

    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}",
              file=sys.stderr)
        sys.exit(1)

    complex_filter = set(args.complexes) if args.complexes else None

    rows = build_rows(args.results_dir, CRITERION, SCOPE,
                      complex_filter=complex_filter)

    print_terminal_table(rows, CRITERION, SCOPE)

    # Resolve output paths
    suffix = _filename_suffix(CRITERION, SCOPE)
    csv_path = args.csv
    md_path = args.md
    if args.auto_names:
        if csv_path is None:
            csv_path = os.path.join(
                SCRIPT_DIR,
                f"unbinding_summary_across_complexes_{suffix}.csv",
            )
        if md_path is None:
            md_path = os.path.join(
                SCRIPT_DIR,
                f"unbinding_summary_across_complexes_{suffix}.md",
            )

    if csv_path:
        write_csv(rows, csv_path)
    if md_path:
        write_markdown(rows, CRITERION, SCOPE, md_path)


if __name__ == "__main__":
    main()
