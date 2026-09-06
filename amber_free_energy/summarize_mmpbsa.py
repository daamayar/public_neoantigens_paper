#!/usr/bin/env python3
"""
Summarize MM-PBSA binding free energy results across all complexes.

Reads the per-complex analysis JSON files from the results/ directory and
generates a summary table with the MM-PBSA enthalpy and the quasi-harmonic
(QH/RRHO) entropy correction.

Two free energy estimates are reported for each complex:
  - dH:    MM-PBSA enthalpy only (no entropy correction)
  - dG_QH: dH + (-TdS)_QH  (quasi-harmonic)

Sign convention: All TdS values stored in JSON are -TdS (positive = unfavorable
entropy penalty). Therefore dG = dH + TdS_stored = dH + (-TdS) = dH - TdS.

For neo/wt pairs, ddG values (neo - wt) are also computed.

Output formats:
  - Terminal: colored table with ANSI codes
  - CSV:      machine-readable (mmpbsa_summary.csv)
  - Markdown: for documentation  (mmpbsa_summary.md)

Usage:
    # Default: summarize all complexes in results/, 500ns window
    python summarize_mmpbsa.py

    # Specify analysis window (100ns, 200ns, 300ns, 400ns, or 500ns)
    python summarize_mmpbsa.py --window 100ns
    python summarize_mmpbsa.py --window 300ns

    # Specify results directory
    python summarize_mmpbsa.py --results-dir /path/to/results

    # Write CSV and Markdown outputs
    python summarize_mmpbsa.py --csv mmpbsa_summary.csv --md mmpbsa_summary.md

    # Summarize only specific complexes
    python summarize_mmpbsa.py --complexes flt3_d835y_neo_a_0201_yimsdsnyv flt3_d835y_wt_a_0201_dimsdsnyv

    # Typical command to use : 
    python summarize_mmpbsa.py --window 100ns --csv mmpbsa_summary_100ns.csv --md mmpbsa_summary_100ns.md
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

# Known neo/wt pairs: (neo_dir_name, wt_dir_name, display_label)
# Order defines the output table row ordering.
PAIRS = [
    ("flt3_d835y_neo_a_0201_yimsdsnyv",    "flt3_d835y_wt_a_0201_dimsdsnyv",     "FLT3 D835Y – A*02:01 – YIMSDSNYV / DIMSDSNYV"),
    ("pik3ca_e545k_neo_a_1101_strdplseitk", "pik3ca_e545k_wt_a_1101_strdplseite", "PIK3CA E545K – A*11:01 – STRDPLSEITK / STRDPLSEITE"),
    ("kras_g12c_neo_a_1101_vvvgacgvgk",    "kras_g12c_wt_a_1101_vvvgaggvgk",     "KRAS G12C – A*11:01 – VVVGACGVGK / VVVGAGGVGK"),
    ("kras_g12d_neo_c_0802_gadgvgksa",     "kras_g12d_wt_c_0802_gaggvgksa",      "KRAS G12D – C*08:02 – GADGVGKSA / GAGGVGKSA"),
    ("kras_g12v_neo_a_1101_vvgavgvgk",     "kras_g12v_wt_a_1101_vvgaggvgk",      "KRAS G12V – A*11:01 – VVGAVGVGK / VVGAGGVGK"),
    ("p53_r175h_neo_a_0201_hmtevvrhc",     "p53_r175h_wt_a_0201_hmtevvrrc",      "p53 R175H – A*02:01 – HMTEVVRHC / HMTEVVRRC"),
    ("pik3ca_h1047l_neo_a_0301_alhggwttk", "pik3ca_h1047l_wt_a_0301_ahhggwttk",  "PIK3CA H1047L – A*03:01 – ALHGGWTTK / AHHGGWTTK"),
]

# Unpaired complexes (no matching wt)
UNPAIRED = [
    ("apc_neo_a_0201_lqmdflvhpa",  "APC neo – A*02:01 – LQMDFLVHPA"),
    ("npm_neo_a_0201_claveevsl",   "NPM neo – A*02:01 – CLAVEEVSL"),
    ("tgfbrii_neo_a_0201_rlsscvpva", "TGFBRii neo – A*02:01 – RLSSCVPVA"),
]

# Column specifications for the summary tables. Each entry is
# (header_label, mean_key, sem_key); the renderers iterate these.
BASE_COLUMNS = [
    ("dH (TOTAL)", "dH_mean",    "dH_sem"),
    ("dG_QH",      "dG_qh_mean", "dG_qh_sem"),
]

# (mean_key, sem_key) pairs for which build_rows() computes ddG (neo - wt).
DDG_KEYS = [
    ("dH_mean", "dH_sem"),
    ("dG_qh_mean", "dG_qh_sem"),
]


def load_complex_data(results_dir, complex_name, window):
    """
    Load analysis results for a single complex from its ensemble summary JSON.

    Args:
        results_dir (str): Path to the top-level results directory.
        complex_name (str): Name of the complex (subdirectory under results_dir).
        window (str): Analysis window, e.g. "500ns" or "100ns".

    Returns:
        dict or None: Dictionary with keys:
            - n_replicas (int): Number of replicas analyzed.
            - dH_mean (float): Ensemble mean of TOTAL enthalpy (kcal/mol).
            - dH_sem (float): SEM of TOTAL enthalpy.
            - qh_TdS_mean (float): Mean -TdS from Quasi-Harmonic method.
            - qh_TdS_sem (float): SEM of QH -TdS.
            - dG_qh_mean (float): Mean dG = dH - TdS_QH.
            - dG_qh_sem (float): SEM of dG_QH (per-replica, or propagated fallback).
            Returns None if data file not found or incomplete.
    """
    json_path = os.path.join(results_dir, complex_name,
                             f"ensemble_summary_{window}.json")
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    n_replicas = data.get("n_replicas", 0)
    if n_replicas == 0:
        return None

    # --- Enthalpy (dH) ---
    components = data.get("components", {})
    total = components.get("TOTAL", {})
    dH_mean = total.get("grand_mean", 0.0)
    dH_sem = total.get("sem", 0.0)

    # --- QHA: Quasi-Harmonic ---
    qha = data.get("qha_ensemble", {})

    qh = qha.get("quasi_harmonic", {})
    qh_TdS_mean = qh.get("TdS_mean", None)
    qh_TdS_sem = qh.get("TdS_sem", None)

    # Compute dG for QH: dG = dH + TdS_stored
    # Convention: TdS_stored = -TdS (positive = unfavorable entropy penalty).
    # So dG = dH + TdS_stored = dH + (-TdS) = dH - TdS (standard thermodynamics).
    #
    # Prefer per-replica dG from ensemble JSON (captures dH-TdS correlation,
    # giving a more accurate SEM). Fall back to propagated SEM for old data.
    import math

    def _compute_dG_propagated(dH_m, dH_s, TdS_m, TdS_s):
        """Compute dG = dH + TdS_stored and propagate SEM assuming independence.

        TdS_stored represents -TdS (positive = unfavorable entropy penalty),
        so dG = dH + TdS_stored = dH - TdS (standard Gibbs equation).

        Note: this fallback overestimates SEM because dH and TdS are correlated.
        The preferred path uses per-replica dG values computed in analyze_results.py.
        """
        if TdS_m is None or TdS_s is None:
            return None, None
        dG = dH_m + TdS_m
        dG_s = math.sqrt(dH_s**2 + TdS_s**2)
        return dG, dG_s

    # QH: use per-replica dG if available
    qh_sem_method = "per_replica"
    if qh.get("dG_mean") is not None:
        dG_qh_mean = qh["dG_mean"]
        dG_qh_sem = qh.get("dG_sem", 0.0)
    else:
        dG_qh_mean, dG_qh_sem = _compute_dG_propagated(
            dH_mean, dH_sem, qh_TdS_mean, qh_TdS_sem)
        qh_sem_method = "propagated"

    return {
        "n_replicas": n_replicas,
        "dH_mean": dH_mean,
        "dH_sem": dH_sem,
        "qh_TdS_mean": qh_TdS_mean,
        "qh_TdS_sem": qh_TdS_sem,
        "dG_qh_mean": dG_qh_mean,
        "dG_qh_sem": dG_qh_sem,
        "qh_sem_method": qh_sem_method,
    }


def fmt_val(mean, sem, width=7):
    """
    Format a value as 'mean +/- sem' string, or 'N/A' if missing.

    Args:
        mean (float or None): Mean value.
        sem (float or None): Standard error of the mean.
        width (int): Minimum width for the mean field.

    Returns:
        str: Formatted string, e.g. ' -70.57 +/-  1.46'.
    """
    if mean is None or sem is None:
        return f"{'N/A':>{width}} +/- {'N/A':>5}"
    return f"{mean:>{width}.2f} +/- {sem:>5.2f}"


def compute_ddG(neo_data, wt_data, key_mean, key_sem):
    """
    Compute ddG = neo - wt and propagate SEM.

    Args:
        neo_data (dict): Results dict for neoantigen complex.
        wt_data (dict): Results dict for wildtype complex.
        key_mean (str): Key for the mean value in the data dict.
        key_sem (str): Key for the SEM value in the data dict.

    Returns:
        tuple: (ddG_mean, ddG_sem) or (None, None) if data missing.
    """
    import math
    neo_m = neo_data.get(key_mean)
    wt_m = wt_data.get(key_mean)
    neo_s = neo_data.get(key_sem)
    wt_s = wt_data.get(key_sem)
    if neo_m is None or wt_m is None or neo_s is None or wt_s is None:
        return None, None
    ddG = neo_m - wt_m
    ddG_sem = math.sqrt(neo_s**2 + wt_s**2)
    return ddG, ddG_sem


def print_terminal_table(all_rows, window, columns=BASE_COLUMNS,
                          title="MM-PBSA Summary", show_sem_note=True):
    """
    Print a formatted summary table to the terminal with ANSI colors.

    Args:
        all_rows (list): List of row dicts with keys: label, type ('neo', 'wt',
            'unpaired', 'ddG', 'separator'), data (dict from load_complex_data
            or ddG values).
        window (str): Analysis window label, e.g. "500ns".
        columns (list): list of (header_label, mean_key, sem_key) tuples
            selecting which value columns to display (default BASE_COLUMNS).
        title (str): table title line.
        show_sem_note (bool): if True, print the propagated-SEM footnote when
            any complex falls back to propagated SEM.
    """
    # ANSI color codes
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    # Key used to colour ddG rows (favourable = negative): the first dG column.
    color_key = columns[1][1] if len(columns) > 1 else columns[0][1]

    cols_hdr = "  ".join(f"{lbl:>20s}" for lbl, _, _ in columns)
    header = f"{'Complex':<50s} {'Reps':>4s}  {cols_hdr}"
    sep_line = "-" * len(header)

    print()
    print(f"{BOLD}{title} — {window} window{RESET}")
    print(f"{BOLD}All values in kcal/mol (mean +/- SEM){RESET}")
    print(sep_line)
    print(f"{BOLD}{header}{RESET}")
    print(sep_line)

    for row in all_rows:
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
            print(f"  {label:<48s} {'—':>4s}  {'NO DATA':>20s}")
            continue

        cells = "  ".join(
            f"{fmt_val(d.get(mk), d.get(sk)):>20s}" for _, mk, sk in columns)

        if rtype == "ddG":
            cval = d.get(color_key)
            color = GREEN if (cval is not None and cval < 0) else RED
            print(f"  {color}{'  ddG (neo-wt)':<48s} {'':>4s}  {cells}{RESET}")
        else:
            # Regular data row
            n = d["n_replicas"]
            warn = f" {YELLOW}(!){RESET}" if n < 10 else ""
            prefix = "  " if rtype in ("neo", "wt") else ""
            print(f"{prefix}{label:<48s} {n:>4d}  {cells}{warn}")

    print(sep_line)

    # Check if any complex uses propagated SEM (fallback for old data)
    if show_sem_note:
        has_propagated = any(
            row.get("data", {}).get("qh_sem_method") == "propagated"
            for row in all_rows if row.get("data")
        )
        if has_propagated:
            print(f"{YELLOW}Note: Some complexes use propagated SEM (no per-replica dG). "
                  f"This overestimates uncertainty vs. per-replica SEM.{RESET}")
    print()


def write_csv(all_rows, window, output_path):
    """
    Write the summary table to a CSV file.

    Args:
        all_rows (list): List of row dicts (same format as print_terminal_table).
        window (str): Analysis window label.
        output_path (str): Output CSV file path.
    """
    fieldnames = [
        "complex", "type", "n_replicas",
        "dH_mean", "dH_sem",
        "qh_TdS_mean", "qh_TdS_sem",
        "dG_qh_mean", "dG_qh_sem",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_rows:
            if row["type"] in ("separator", "section_header"):
                continue
            d = row.get("data")
            if d is None:
                writer.writerow({"complex": row["label"], "type": row["type"]})
                continue

            csv_row = {
                "complex": row["label"],
                "type": row["type"],
            }
            for key in fieldnames[2:]:
                csv_row[key] = d.get(key, "")
            writer.writerow(csv_row)

    print(f"CSV written to {output_path}")


def _md_section(f, all_rows, columns, heading):
    """
    Write one Markdown table section (header + rows) for the given column spec.

    Args:
        f: open file handle.
        all_rows (list): row dicts (see build_rows()).
        columns (list): (header_label, mean_key, sem_key) tuples.
        heading (str): section heading (## level).
    """
    def _mdfmt(mean, sem):
        if mean is None or sem is None:
            return "N/A"
        return f"{mean:.2f} +/- {sem:.2f}"

    col_hdr = " | ".join(lbl for lbl, _, _ in columns)
    col_sep = "|".join("-" * (len(lbl) + 2) for lbl, _, _ in columns)
    table_head = f"| Complex | Reps | {col_hdr} |\n"
    table_sep = f"|---------|------|{col_sep}|\n"

    f.write(f"## {heading}\n\n")
    f.write(table_head)
    f.write(table_sep)

    for row in all_rows:
        if row["type"] in ("separator",):
            continue
        if row["type"] == "section_header":
            f.write(f"\n**{row['label']}**\n\n")
            f.write(table_head)
            f.write(table_sep)
            continue

        d = row.get("data")
        label = row["label"]

        if d is None:
            empties = " | ".join("" for _ in columns)
            f.write(f"| {label} | — | NO DATA | {empties} |\n")
            continue

        n = d.get("n_replicas", "")
        if row["type"] == "ddG":
            n = ""

        cells = " | ".join(_mdfmt(d.get(mk), d.get(sk)) for _, mk, sk in columns)
        f.write(f"| {label} | {n} | {cells} |\n")

    f.write("\n")


def write_markdown(all_rows, window, output_path):
    """
    Write the summary table as a Markdown file.

    Args:
        all_rows (list): List of row dicts.
        window (str): Analysis window label.
        output_path (str): Output Markdown file path.
    """
    with open(output_path, "w") as f:
        f.write(f"# MM-PBSA Binding Free Energy Summary ({window})\n\n")
        f.write("All values in kcal/mol (mean +/- SEM).\n\n")

        _md_section(f, all_rows, BASE_COLUMNS, "Per-Complex Results")

        f.write("---\n")
        f.write("*Generated by `summarize_mmpbsa.py`*\n")

    print(f"Markdown written to {output_path}")


def build_rows(results_dir, window):
    """
    Build the list of display rows from all available complex results.

    Loads data for each known neo/wt pair and unpaired complex, computes
    ddG values, and organizes into a structured row list for display.

    Args:
        results_dir (str): Path to the results directory.
        window (str): Analysis window, e.g. "500ns".

    Returns:
        list: List of row dicts. Each has keys:
            - label (str): Display label for the row.
            - type (str): One of 'neo', 'wt', 'ddG', 'unpaired',
              'separator', 'section_header'.
            - data (dict or None): Results data or ddG values.
    """
    rows = []

    # --- Paired complexes ---
    rows.append({"label": "Paired Complexes (neo vs wt)", "type": "section_header"})

    for neo_name, wt_name, pair_label in PAIRS:
        neo_data = load_complex_data(results_dir, neo_name, window)
        wt_data = load_complex_data(results_dir, wt_name, window)

        # Extract short peptide names from pair_label
        parts = pair_label.split(" – ")
        hla = parts[0] if len(parts) > 0 else ""
        peptides = parts[-1] if len(parts) > 1 else ""
        mutation = parts[1] if len(parts) > 2 else ""
        prefix = f"{mutation} – " if mutation else ""

        neo_pep = peptides.split(" / ")[0] if " / " in peptides else neo_name
        wt_pep = peptides.split(" / ")[1] if " / " in peptides else wt_name

        neo_label = f"{prefix}{hla} – {neo_pep} (neo)"
        wt_label = f"{prefix}{hla} – {wt_pep} (wt)"

        rows.append({"label": neo_label, "type": "neo", "data": neo_data})
        rows.append({"label": wt_label, "type": "wt", "data": wt_data})

        # Compute ddG if both are available
        if neo_data is not None and wt_data is not None:
            ddG_data = {}
            for mean_key, sem_key in DDG_KEYS:
                ddm, dds = compute_ddG(neo_data, wt_data, mean_key, sem_key)
                ddG_data[mean_key] = ddm
                ddG_data[sem_key] = dds
            rows.append({"label": f"  ddG (neo-wt)", "type": "ddG",
                         "data": ddG_data})

        rows.append({"label": "", "type": "separator"})

    # --- Unpaired complexes ---
    rows.append({"label": "Unpaired Complexes", "type": "section_header"})

    for name, label in UNPAIRED:
        data = load_complex_data(results_dir, name, window)
        rows.append({"label": label, "type": "unpaired", "data": data})

    # --- Auto-detect any complexes not in PAIRS or UNPAIRED ---
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
            if d not in known:
                data = load_complex_data(results_dir, d, window)
                if data is not None:
                    extra.append({"label": d, "type": "unpaired", "data": data})

    if extra:
        rows.append({"label": "", "type": "separator"})
        rows.append({"label": "Other Complexes (auto-detected)",
                      "type": "section_header"})
        rows.extend(extra)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Summarize MM-PBSA binding free energy results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default summary (500ns)
    python summarize_mmpbsa.py

    # 100ns window
    python summarize_mmpbsa.py --window 100ns

    # Export CSV and Markdown
    python summarize_mmpbsa.py --csv mmpbsa_summary.csv --md mmpbsa_summary.md

    # Multiple windows (run separately for each)
    python summarize_mmpbsa.py --window 500ns
    python summarize_mmpbsa.py --window 300ns
    python summarize_mmpbsa.py --window 100ns
        """,
    )
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--window", type=str, default="500ns",
                        choices=["100ns", "200ns", "300ns", "400ns", "500ns"],
                        help="Analysis window (default: 500ns)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Output CSV file path")
    parser.add_argument("--md", type=str, default=None,
                        help="Output Markdown file path")

    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)

    rows = build_rows(args.results_dir, args.window)

    print_terminal_table(rows, args.window)

    # Optional file outputs
    if args.csv:
        write_csv(rows, args.window, args.csv)
    if args.md:
        write_markdown(rows, args.window, args.md)


if __name__ == "__main__":
    main()
