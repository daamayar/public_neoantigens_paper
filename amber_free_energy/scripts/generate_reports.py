#!/usr/bin/env python3
"""
Phase 5: Report Generation
Generate the Markdown analysis report.

The report contains:
  - System description and methods
  - Ensemble energy components with full statistics
  - Per-replica energy breakdown
  - Binding free energy, enthalpy-only and quasi-harmonic entropy-corrected
  - Convergence assessment (cumulative mean, block averaging,
    inter-replica consistency, autocorrelation)
  - Validation checklist
  - Convergence across analysis windows (100ns-500ns)

Usage:
    python -m scripts.generate_reports <complex_name>
"""
import os
import json
import argparse
from datetime import datetime

from . import config
from . import utils


def load_analysis_data(complex_name):
    """Load all analysis results from JSON files."""
    results_dir = utils.get_results_dir(complex_name)

    data = {}
    for fname in os.listdir(results_dir):
        if fname.endswith(".json"):
            key = fname.replace(".json", "")
            data[key] = utils.load_metadata(os.path.join(results_dir, fname))

    # Load topology metadata from first available replica
    for rep in range(1, config.NUM_REPLICAS + 1):
        meta_path = os.path.join(
            utils.get_mmpbsa_workdir(complex_name, rep, config.REFERENCE_WINDOW),
            "topology_metadata.json"
        )
        if os.path.exists(meta_path):
            data["topology_metadata"] = utils.load_metadata(meta_path)
            break

    return data


def _binding_fe_summary(ens):
    """
    Build the binding free energy summary rows.

    Returns a list of dicts keyed: approach, dH, dH_sem, TdS, TdS_sem, dG, dG_sem.
    """
    rows = []
    total = ens.get("components", {}).get("TOTAL", {})
    dH = total.get("grand_mean", 0.0)
    dH_sem = total.get("sem", 0.0)

    rows.append({
        "approach": "Enthalpy-only (dH)",
        "dH": dH, "dH_sem": dH_sem,
        "TdS": None, "TdS_sem": None,
        "dG": dH, "dG_sem": dH_sem,
    })

    if ens.get("qha_ensemble"):
        qh = ens["qha_ensemble"]["quasi_harmonic"]
        # Prefer the per-replica dG, which captures the dH-TdS correlation; fall
        # back to a propagated SEM for older data that lacks the per-replica values.
        if "dG_mean" in qh:
            dG_qh, dG_qh_sem, sem_method = qh["dG_mean"], qh["dG_sem"], "per_replica"
        else:
            dG_qh = dH + qh["TdS_mean"]
            dG_qh_sem = (dH_sem ** 2 + qh["TdS_sem"] ** 2) ** 0.5
            sem_method = "propagated"
        rows.append({
            "approach": "QHA - Quasi-Harmonic (RRHO)",
            "dH": dH, "dH_sem": dH_sem,
            "TdS": qh["TdS_mean"], "TdS_sem": qh["TdS_sem"],
            "dG": dG_qh, "dG_sem": dG_qh_sem,
            "sem_method": sem_method,
        })

    return rows


def _per_replica_all_approaches(ens):
    """
    Build the per-replica table.

    Returns a list of dicts keyed: replica, dH, TdS_qh, dG_qh.
    """
    rows = []
    qha_results = ens.get("qha_results", [])
    qha_map = {qr["replica"]: qr for qr in qha_results}
    qha_order = [qr["replica"] for qr in qha_results]

    # Per-replica dG from the ensemble, which records the exact dH used during
    # analysis; recomputing it here from a different code path could diverge.
    qh_dG_list = ens.get("qha_ensemble", {}).get("quasi_harmonic", {}).get("per_replica_dG", [])

    for r in ens.get("per_replica", []):
        rep = r["replica"]
        dH = r.get("TOTAL", 0.0)
        row = {"replica": rep, "dH": dH}
        if rep in qha_map:
            idx = qha_order.index(rep)
            row["TdS_qh"] = qha_map[rep]["quasi_harmonic"]["TdS"]
            row["dG_qh"] = (qh_dG_list[idx] if 0 <= idx < len(qh_dG_list)
                            else dH + qha_map[rep]["quasi_harmonic"]["TdS"])
        rows.append(row)
    return rows


# =============================================================================
# MARKDOWN REPORT
# =============================================================================

def generate_markdown_report(complex_name, data):
    """Generate a comprehensive Markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append(f"# MM-PBSA Binding Free Energy Report: {complex_name}")
    lines.append(f"\n**Generated**: {now}")
    lines.append(f"**Pipeline**: AMBER MMPBSA.py with CHARMM36m force field")
    lines.append(f"**Replicas**: {config.NUM_REPLICAS}")
    lines.append(f"**Simulation time**: {config.TOTAL_SIM_TIME_NS} ns per replica")
    lines.append("")

    # Section 1: System description
    lines.append("## 1. System Description")
    lines.append("")
    if "topology_metadata" in data:
        meta = data["topology_metadata"]
        lines.append(f"- **Complex**: {meta['complex_name']}")
        lines.append(f"- **Solvated atoms**: {meta['solvated_atoms']:,}")
        lines.append(f"- **Dry complex atoms**: {meta['complex_atoms']:,}")
        lines.append(f"- **Receptor atoms**: {meta['receptor_atoms']:,} (HLA alpha + beta-2-microglobulin)")
        lines.append(f"- **Ligand atoms**: {meta['ligand_atoms']:,} (peptide)")
        lines.append(f"- **Receptor mask**: {meta['receptor_mask']}")
        lines.append(f"- **Ligand mask**: {meta['ligand_mask']}")
        lines.append(f"- **CMAP terms**: {meta['cmap_terms']}")
        lines.append(f"- **Box dimensions**: {meta['box_dims'][0]:.1f} x {meta['box_dims'][1]:.1f} x {meta['box_dims'][2]:.1f} A")
        lines.append("")

    # Section 2: Methods
    lines.append("## 2. Methods")
    lines.append("")
    lines.append("### 2.1 Topology Conversion")
    lines.append("- CHARMM PSF/PDB converted to AMBER ChamberParm prmtop using ParmEd Python API")
    lines.append("- Force field: CHARMM36m protein parameters with CMAP corrections")
    lines.append(f"- Parameter files: {', '.join(os.path.basename(f) for f in config.PARAM_FILES)}")
    lines.append("")

    lines.append("### 2.2 Trajectory Preparation")
    lines.append("- Trajectories pre-processed with cpptraj")
    lines.append(f"- Total simulation: {config.TOTAL_SIM_TIME_NS} ns ({config.TOTAL_FRAMES} frames)")
    lines.append(f"- Equilibration excluded: first {config.EQUILIBRATION_NS} ns ({config.EQUILIBRATION_FRAMES} frames)")
    lines.append("")
    lines.append("| Analysis Window | Frame Range | Stride | Frames/Replica |")
    lines.append("|:-:|:-:|:-:|:-:|")
    for wname, wcfg in config.ANALYSIS_WINDOWS.items():
        n_frames = (wcfg["last_frame"] - wcfg["first_frame"] + 1) // wcfg["stride"]
        lines.append(f"| {wname} | {wcfg['first_frame']}-{wcfg['last_frame']} | {wcfg['stride']} | ~{n_frames} |")
    lines.append("")

    lines.append("### 2.3 MMPBSA Parameters")
    lines.append("")
    p = config.MMPBSA_PARAMS
    lines.append("| Parameter | Value | Rationale |")
    lines.append("|:-:|:-:|---|")
    lines.append(f"| radiopt | {p['radiopt']} | Mandatory for CHARMM (optimized radii only for AMBER) |")
    lines.append(f"| inp | {p['inp']} | Mandatory when radiopt=0 |")
    lines.append(f"| use_sander | {p['use_sander']} | Required for CMAP/Urey-Bradley/NBFIX |")
    lines.append(f"| indi | {p['indi']} | Interior dielectric |")
    lines.append(f"| exdi | {p['exdi']} | Exterior dielectric (water) |")
    lines.append(f"| istrng | {p['istrng']} | Ionic strength (150 mM) |")
    lines.append(f"| temperature | {p['temperature']} | Simulation temperature |")
    lines.append(f"| PB radii | mbondi_pb3 | Element-based PB radii (assigned via ParmEd changeRadii) |")
    lines.append("")

    lines.append("### 2.4 Entropy Correction Methods")
    lines.append("")
    lines.append("The binding free energy is reported two ways, both computed as post-processing "
                 "from the enthalpy-only MMPBSA results:")
    lines.append("")
    lines.append("1. **Enthalpy-only (dH)**: no entropy correction, dG = dH.")
    lines.append("2. **Quasi-Harmonic Analysis (QHA)**: configurational entropy from the "
                 "mass-weighted covariance matrix of the CA coordinates, evaluated with the "
                 "quantum harmonic oscillator partition function "
                 "(omega_i = sqrt(kBT/lambda_i), alpha_i = h_bar*omega_i/(kBT)), giving "
                 "-TdS = -T [S(complex) - S(receptor) - S(ligand)].")
    lines.append("")

    # Sections 3+: Results for each window (dynamic numbering)
    section_num = 3
    for window in config.ANALYSIS_WINDOWS:
        ens_key = f"ensemble_summary_{window}"
        if ens_key not in data:
            continue

        ens = data[ens_key]
        lines.append(f"## {section_num}. Results ({window})")
        lines.append("")

        # N.1 Ensemble energy components
        lines.append(f"### {section_num}.1 Ensemble Energy Components ({window})")
        lines.append(f"\n**Number of replicas**: {ens['n_replicas']}")
        lines.append("")
        lines.append("| Component | Grand Mean | SEM | SD (between) | SD (within) | Total SD |")
        lines.append("|:-:|:-:|:-:|:-:|:-:|:-:|")
        for comp in ["VDWAALS", "EEL", "EPB", "ENPOLAR", "TOTAL"]:
            if comp in ens.get("components", {}):
                c = ens["components"][comp]
                lines.append(f"| {comp} | {c['grand_mean']:.2f} | {c['sem']:.2f} | "
                             f"{c['sd_between']:.2f} | {c['sd_within']:.2f} | {c['total_sd']:.2f} |")
        lines.append("")

        # N.2 Per-replica energy components
        lines.append(f"### {section_num}.2 Per-Replica Energy Components ({window})")
        lines.append("")
        components = ["VDWAALS", "EEL", "EPB", "ENPOLAR", "TOTAL"]
        available_comps = [c for c in components if any(c in r for r in ens.get("per_replica", []))]
        header = "| Replica | " + " | ".join(available_comps) + " |"
        separator = "|:-:|" + ":-:|" * len(available_comps)
        lines.append(header)
        lines.append(separator)
        for r in ens.get("per_replica", []):
            vals = " | ".join(f"{r.get(c, 'N/A'):.2f}" if isinstance(r.get(c), (int, float)) else "N/A"
                              for c in available_comps)
            lines.append(f"| rep_{r['replica']} | {vals} |")
        lines.append("")

        # N.3 Binding Free Energy Summary (all 3 approaches)
        lines.append(f"### {section_num}.3 Binding Free Energy Summary ({window})")
        lines.append("")
        lines.append("All values in kcal/mol (mean +/- SEM).")
        lines.append("")
        lines.append("| Approach | dH (enthalpy) | -TdS (entropy) | dG (binding) |")
        lines.append("|---|:-:|:-:|:-:|")
        has_propagated_sem = False
        for row in _binding_fe_summary(ens):
            dH_str = f"{row['dH']:.2f} +/- {row['dH_sem']:.2f}"
            if row["TdS"] is not None:
                TdS_str = f"{row['TdS']:.2f} +/- {row['TdS_sem']:.2f}"
            else:
                TdS_str = "---"
            sem_flag = ""
            if row.get("sem_method") == "propagated":
                sem_flag = " *"
                has_propagated_sem = True
            dG_str = f"{row['dG']:.2f} +/- {row['dG_sem']:.2f}{sem_flag}"
            lines.append(f"| {row['approach']} | {dH_str} | {TdS_str} | {dG_str} |")
        if has_propagated_sem:
            lines.append("")
            lines.append("\\* SEM estimated via error propagation (no per-replica dG available). "
                         "This overestimates uncertainty vs. per-replica SEM.")
        lines.append("")

        # N.4 Per-Replica Binding Free Energy (all approaches)
        lines.append(f"### {section_num}.4 Per-Replica Binding Free Energy ({window})")
        lines.append("")
        per_rep_rows = _per_replica_all_approaches(ens)
        has_qha = any("dG_qh" in r for r in per_rep_rows)

        header_parts = ["Replica", "dH"]
        if has_qha:
            header_parts += ["-TdS(QH)", "dG(QH)"]
        lines.append("| " + " | ".join(header_parts) + " |")
        lines.append("|:-:|" + ":-:|" * (len(header_parts) - 1))

        for r in per_rep_rows:
            vals = [f"rep_{r['replica']}", f"{r['dH']:.2f}"]
            if has_qha:
                vals.append(f"{r.get('TdS_qh', 0):.2f}" if "TdS_qh" in r else "N/A")
                vals.append(f"{r.get('dG_qh', 0):.2f}" if "dG_qh" in r else "N/A")
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

        # N.5 Entropy Correction Statistics
        lines.append(f"### {section_num}.5 Entropy Correction Statistics ({window})")
        lines.append("")
        lines.append("| Method | Mean -TdS | SD | SEM |")
        lines.append("|---|:-:|:-:|:-:|")
        if ens.get("qha_ensemble"):
            qh = ens["qha_ensemble"]["quasi_harmonic"]
            lines.append(f"| Quasi-Harmonic (RRHO) | {qh['TdS_mean']:.2f} "
                         f"| {qh['TdS_std']:.2f} | {qh['TdS_sem']:.2f} |")
        lines.append("")

        section_num += 1

    # Convergence assessment (continues from section_num)
    for window in config.ANALYSIS_WINDOWS:
        conv_key = f"convergence_{window}"
        if conv_key not in data:
            continue

        conv = data[conv_key]
        lines.append(f"## {section_num}. Convergence Assessment ({window})")
        lines.append("")

        # Cumulative mean
        if "cumulative_mean" in conv:
            lines.append(f"### {section_num}.1 Cumulative Mean ({window})")
            lines.append("")
            lines.append(f"![Cumulative mean](plots/convergence_cumulative_{window}.png)")
            lines.append("")
            for rep, cm in sorted(conv["cumulative_mean"].items()):
                lines.append(f"- rep_{rep}: final mean = {cm['final_mean']:.2f} kcal/mol ({cm['n_frames']} frames)")
            lines.append("")

        # Block averaging
        if "block_averaging" in conv:
            lines.append(f"### {section_num}.2 Block Averaging ({window})")
            lines.append("")
            lines.append(f"![Block averaging](plots/convergence_blocks_{window}.png)")
            lines.append("")
            for rep, ba in sorted(conv["block_averaging"].items()):
                lines.append(f"- rep_{rep}: block spread = {ba['block_spread']:.2f} kcal/mol, "
                             f"blocks = {ba['block_means']}")
            lines.append("")

        # Inter-replica
        if "inter_replica" in conv:
            lines.append(f"### {section_num}.3 Inter-Replica Consistency ({window})")
            lines.append("")
            ir = conv["inter_replica"]
            if "replicas" in ir:
                outliers = [r for r in ir["replicas"] if r.get("is_outlier")]
                if outliers:
                    lines.append(f"**Outliers detected**: {[r['replica'] for r in outliers]}")
                else:
                    lines.append("No outliers detected (all within 2 sigma).")
            lines.append("")

        # Autocorrelation
        if "autocorrelation" in conv:
            lines.append(f"### {section_num}.4 Autocorrelation Analysis ({window})")
            lines.append("")
            lines.append(f"![Autocorrelation](plots/convergence_acf_{window}.png)")
            lines.append("")
            lines.append("| Replica | Correlation Time (frames) | Effective N | Total N |")
            lines.append("|:-:|:-:|:-:|:-:|")
            for rep, ac in sorted(conv["autocorrelation"].items()):
                lines.append(f"| rep_{rep} | {ac['correlation_time']:.1f} | "
                             f"{ac['effective_n']:.0f} | {ac['n_frames']} |")
            lines.append("")

        section_num += 1

    # Validation
    for window in config.ANALYSIS_WINDOWS:
        ens_key = f"ensemble_summary_{window}"
        if ens_key not in data:
            continue

        lines.append(f"## {section_num}. Validation Checklist ({window})")
        lines.append("")

        ens = data[ens_key]
        if "components" in ens:
            if "TOTAL" in ens["components"]:
                total = ens["components"]["TOTAL"]["grand_mean"]
                status = "PASS" if total < 0 else "FAIL"
                lines.append(f"- [{status}] Sign of dG_total: {total:.2f} kcal/mol")

                sd = ens["components"]["TOTAL"]["sd_between"]
                status = "PASS" if sd < abs(total) else "FAIL"
                lines.append(f"- [{status}] SD vs mean: SD={sd:.2f}, |mean|={abs(total):.2f}")

            if "EEL" in ens["components"] and "EPB" in ens["components"]:
                eel = ens["components"]["EEL"]["grand_mean"]
                epb = ens["components"]["EPB"]["grand_mean"]
                status_eel = "PASS" if eel < 0 else "FAIL"
                status_epb = "PASS" if epb > 0 else "FAIL"
                lines.append(f"- [{status_eel}] EEL is negative: {eel:.2f} kcal/mol")
                lines.append(f"- [{status_epb}] EPB is positive: {epb:.2f} kcal/mol")
                lines.append(f"- [INFO] Net electrostatic (EEL+EPB): {eel+epb:.2f} kcal/mol")

            if "VDWAALS" in ens["components"]:
                vdw = ens["components"]["VDWAALS"]["grand_mean"]
                in_range = -80 <= vdw <= -40
                status = "PASS" if in_range else "WARNING"
                lines.append(f"- [{status}] VDW magnitude: {vdw:.2f} kcal/mol (expected -80 to -40)")

        lines.append("")
        section_num += 1

    # Section: Convergence across analysis windows
    conv_data = data.get("convergence_across_windows")
    if conv_data:
        lines.append("## Convergence Across Analysis Windows")
        lines.append("")

        # Summary table
        lines.append("### Ensemble dG by Window")
        lines.append("")
        ref = config.REFERENCE_WINDOW
        lines.append(f"| Window | dG (mean +/- SEM) | n_replicas | Diff vs {ref} | Significant? |")
        lines.append("|:-:|:-:|:-:|:-:|:-:|")
        for w in conv_data.get("windows", []):
            s = conv_data["convergence_summary"].get(w, {})
            mean_str = f"{s['mean']:.2f}" if s.get("mean") is not None else "N/A"
            sem_str = f"+/- {s['sem']:.2f}" if s.get("sem") is not None else ""
            pw = conv_data.get("pairwise_vs_reference", {}).get(w, {})
            if w == ref:
                diff_str = "---"
                sig_str = "---"
            elif pw.get("difference") is not None:
                diff_str = f"{pw['difference']:.2f}"
                if pw.get("sem_diff") is not None:
                    diff_str += f" +/- {pw['sem_diff']:.2f}"
                sig_str = "Yes" if pw.get("significant") else "No"
            else:
                diff_str = "N/A"
                sig_str = "N/A"
            lines.append(f"| {w} | {mean_str} {sem_str} | {s.get('n_replicas', 'N/A')} | "
                         f"{diff_str} | {sig_str} |")
        lines.append("")

        # Convergence plot
        lines.append("### Convergence Plot")
        lines.append("")
        lines.append("![Convergence](plots/convergence_across_windows.png)")
        lines.append("")

        # Assessment
        lines.append("### Assessment")
        lines.append("")
        assessment = conv_data.get("convergence_assessment", "")
        if assessment:
            lines.append(assessment)
        lines.append("")

    # Section: Conclusions
    lines.append("## Conclusions")
    lines.append("")
    ref_key = f"ensemble_summary_{config.REFERENCE_WINDOW}"
    if ref_key in data and "TOTAL" in data[ref_key].get("components", {}):
        ens_ref = data[ref_key]
        lines.append(f"The MM-PBSA binding free energy for {complex_name} ({config.REFERENCE_WINDOW}, "
                     f"{ens_ref['n_replicas']} replicas):")
        lines.append("")
        lines.append("| Approach | dG (kcal/mol) |")
        lines.append("|---|:-:|")
        for row in _binding_fe_summary(ens_ref):
            lines.append(f"| **{row['approach']}** | **{row['dG']:.2f} +/- {row['dG_sem']:.2f}** |")
        lines.append("")
        lines.append("Note: MM-PBSA systematically overestimates absolute binding strength. "
                     "These values are suitable for relative ranking of peptide variants, "
                     "not for comparison with experimental absolute binding free energies.")
    lines.append("")

    return "\n".join(lines)


def generate_reports(complex_name):
    """Generate the Markdown report for one complex."""
    print(f"\n{'='*60}")
    print(f"Phase 5: Report generation for {complex_name}")
    print(f"{'='*60}")

    results_dir = utils.get_results_dir(complex_name)
    reports_dir = os.path.join(results_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Load analysis data
    data = load_analysis_data(complex_name)
    if not data:
        print("  ERROR: No analysis data found. Run analysis first.")
        return

    # Markdown report
    print("  Generating Markdown report...")
    md_content = generate_markdown_report(complex_name, data)
    md_path = os.path.join(reports_dir, "report.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"    Saved: {md_path}")

    print(f"\n  Reports saved to {reports_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate analysis reports")
    parser.add_argument("complex_name", help="Complex directory name")
    args = parser.parse_args()
    generate_reports(args.complex_name)


if __name__ == "__main__":
    main()
