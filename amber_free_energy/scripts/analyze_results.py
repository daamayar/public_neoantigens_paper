#!/usr/bin/env python3
"""
Phase 4: Results Analysis
Parse, aggregate, validate, and analyze MMPBSA results across replicas.

Includes:
- Per-replica energy parsing
- Ensemble statistics (mean, SEM, SD)
- 4 convergence assessments (cumulative mean, block averaging,
  inter-replica consistency, autocorrelation)
- Validation checklist
- Convergence across analysis windows (100ns through 500ns)

Usage:
    python -m scripts.analyze_results <complex_name> [--window 100ns|200ns|300ns|400ns|500ns|all]
"""
import os
import json
import argparse
import numpy as np

from . import config
from . import utils
from .compute_entropy import compute_qha_entropy, aggregate_qha_results


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_mmpbsa_output(filepath):
    """
    Parse FINAL_RESULTS_MMPBSA.dat for energy components.

    Returns:
        dict with keys: "complex", "receptor", "ligand", "delta"
        Each is a dict of {component_name: {"mean": float, "std": float, "sem": float}}
    """
    results = {"complex": {}, "receptor": {}, "ligand": {}, "delta": {}}
    current_section = None

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()

            # Detect section headers
            if "GENERALIZED BORN" in stripped or "POISSON BOLTZMANN" in stripped:
                continue
            if stripped.startswith("Complex:"):
                current_section = "complex"
                continue
            elif stripped.startswith("Receptor:"):
                current_section = "receptor"
                continue
            elif stripped.startswith("Ligand:"):
                current_section = "ligand"
                continue
            elif "Differences" in stripped or "DELTA SECTION" in stripped:
                current_section = "delta"
                continue

            if current_section is None:
                continue

            # Parse energy component lines. Formats:
            #   "COMPONENT            MEAN    STD    SEM"     (single-word name)
            #   "DELTA TOTAL          MEAN    STD    SEM"     (multi-word name)
            #   "1-4 VDW              MEAN    STD    SEM"     (multi-word name)
            # Strategy: find the last 2-3 numeric values at end of line
            parts = stripped.split()
            if len(parts) >= 3:
                # Try to parse the last 3 values as floats (mean, std, sem)
                try:
                    sem_val = float(parts[-1])
                    std_val = float(parts[-2])
                    mean_val = float(parts[-3])
                    name = " ".join(parts[:-3])
                    if name:
                        results[current_section][name] = {
                            "mean": mean_val,
                            "std": std_val,
                            "sem": sem_val,
                        }
                except (ValueError, IndexError):
                    # Try last 2 values (mean, std)
                    try:
                        std_val = float(parts[-1])
                        mean_val = float(parts[-2])
                        name = " ".join(parts[:-2])
                        if name:
                            results[current_section][name] = {
                                "mean": mean_val,
                                "std": std_val,
                                "sem": None,
                            }
                    except (ValueError, IndexError):
                        pass

    # Normalize delta section keys: "DELTA TOTAL" -> "TOTAL", "DELTA G gas" -> "G gas"
    normalized_delta = {}
    for key, val in results["delta"].items():
        norm_key = key.replace("DELTA ", "") if key.startswith("DELTA ") else key
        normalized_delta[norm_key] = val
    results["delta"] = normalized_delta

    return results


def _get_nested(data, key):
    """Access a key from an API result using dict-style or attribute-style access."""
    try:
        return data[key]
    except (KeyError, TypeError):
        pass
    try:
        return getattr(data, key)
    except AttributeError:
        pass
    return None


def parse_frame_energies_from_info(rep_dir):
    """
    Parse per-frame energies from _MMPBSA_info using the AMBER API.

    The API requires running from the workdir (it opens relative file paths).
    Supports both dict-style and attribute-style access for compatibility
    with different MMPBSA.py versions.

    Returns:
        dict: {component_name: np.array of per-frame values} or None
    """
    info_file = os.path.join(rep_dir, "_MMPBSA_info")
    if not os.path.exists(info_file):
        print(f"    Warning: _MMPBSA_info not found in {rep_dir}. "
              f"Per-frame energies unavailable (convergence analysis will be skipped).")
        return None

    try:
        from MMPBSA_mods import API as MMPBSAapi
    except ImportError:
        print(f"    Warning: MMPBSA_mods.API not importable. "
              f"Per-frame energies unavailable (convergence analysis will be skipped).")
        return None

    try:
        # API opens files relative to CWD, so we must change directory
        orig_dir = os.getcwd()
        os.chdir(rep_dir)
        try:
            data = MMPBSAapi.load_mmpbsa_info("_MMPBSA_info")
        finally:
            os.chdir(orig_dir)

        # Try dict-style first, then attribute-style access
        pb_data = _get_nested(data, "pb")
        if pb_data is None:
            print(f"    Warning: AMBER API returned data without 'pb' key. "
                  f"Available keys: {list(data.keys()) if hasattr(data, 'keys') else dir(data)}. "
                  f"Per-frame energies unavailable.")
            return None

        frame_data = {}
        sections_found = []
        for section in ["complex", "receptor", "ligand"]:
            section_data = _get_nested(pb_data, section)
            if section_data is not None:
                sections_found.append(section)
                items = section_data.items() if hasattr(section_data, 'items') else []
                for component, values in items:
                    key = f"{section}_{component}"
                    frame_data[key] = np.array(values)

        if len(sections_found) < 3:
            missing = set(["complex", "receptor", "ligand"]) - set(sections_found)
            print(f"    Warning: AMBER API missing sections {missing} in PB data. "
                  f"Delta computation will be skipped.")
        else:
            # Compute delta (complex - receptor - ligand) for each component
            complex_data = _get_nested(pb_data, "complex")
            receptor_data = _get_nested(pb_data, "receptor")
            ligand_data = _get_nested(pb_data, "ligand")
            complex_keys = complex_data.keys() if hasattr(complex_data, 'keys') else []
            for component in complex_keys:
                r_vals = _get_nested(receptor_data, component)
                l_vals = _get_nested(ligand_data, component)
                if r_vals is not None and l_vals is not None:
                    delta = (np.array(_get_nested(complex_data, component))
                             - np.array(r_vals) - np.array(l_vals))
                    frame_data[f"delta_{component}"] = delta

        return frame_data if frame_data else None
    except Exception as e:
        print(f"    Warning: AMBER API parsing failed for {rep_dir}: {type(e).__name__}: {e}. "
              f"Per-frame energies unavailable (convergence analysis will be skipped).")
        return None



# =============================================================================
# ENSEMBLE AGGREGATION
# =============================================================================

def aggregate_replicas(complex_name, analysis_window):
    """
    Aggregate results across all replicas for a given analysis window.

    Returns:
        dict with aggregated statistics
    """
    print(f"\n--- Aggregating {analysis_window} results ---")

    replica_results = []
    all_frame_energies = {}

    for rep in range(1, config.NUM_REPLICAS + 1):
        workdir = utils.get_mmpbsa_workdir(complex_name, rep, analysis_window)
        output_file = os.path.join(workdir, "FINAL_RESULTS_MMPBSA.dat")

        if not os.path.exists(output_file):
            print(f"  WARNING: No output for replica {rep}")
            continue

        # Parse summary results
        parsed = parse_mmpbsa_output(output_file)
        if "TOTAL" not in parsed["delta"]:
            print(f"  WARNING: No TOTAL in delta for replica {rep}")
            continue

        replica_results.append({
            "replica": rep,
            "parsed": parsed,
        })

        print(f"  rep_{rep}: dG = {parsed['delta']['TOTAL']['mean']:.2f} "
              f"+/- {parsed['delta']['TOTAL']['std']:.2f} kcal/mol")

        # Parse per-frame energies
        frame_data = parse_frame_energies_from_info(workdir)
        if frame_data:
            all_frame_energies[rep] = frame_data

    if not replica_results:
        print("  ERROR: No valid results found!")
        return None

    if len(replica_results) == 1:
        print("  WARNING: Only 1 replica has valid results. "
              "Ensemble statistics (SD, SEM) will be zero. "
              "At least 2 replicas are needed for meaningful uncertainty estimates.")

    # Compute ensemble statistics.
    #
    # Hierarchical model: replicas are the independent sampling units.
    #   - var_between: inter-replica variance of per-replica means (ddof=1)
    #   - var_within: mean of per-replica variances (frame-level noise)
    #   - total_sd: sqrt(var_between + var_within), total spread
    #   - SEM: sd_between / sqrt(n), standard error of the ensemble mean
    #
    # SEM uses only var_between because replicas are the independent units.
    # Within-replica variance is averaged out in the per-replica means.
    # This is the standard approach in MM-PBSA ensemble analyses
    # (Genheden & Ryde, Expert Opin. Drug Discov. 2015, 10, 449-461).
    n_reps = len(replica_results)
    components = ["VDWAALS", "EEL", "EPB", "ENPOLAR", "TOTAL"]

    ensemble = {"n_replicas": n_reps, "components": {}, "per_replica": []}

    for comp in components:
        values = []
        stds = []
        for r in replica_results:
            if comp in r["parsed"]["delta"]:
                values.append(r["parsed"]["delta"][comp]["mean"])
                stds.append(r["parsed"]["delta"][comp]["std"])

        if not values:
            continue

        values = np.array(values)
        stds = np.array(stds)

        grand_mean = np.mean(values)
        var_between = np.var(values, ddof=1) if len(values) > 1 else 0.0
        sd_between = np.sqrt(var_between)
        var_within = np.mean(stds ** 2)
        sd_within = np.sqrt(var_within)
        total_sd = np.sqrt(var_between + var_within)
        sem = sd_between / np.sqrt(len(values)) if len(values) > 1 else 0.0

        ensemble["components"][comp] = {
            "grand_mean": float(grand_mean),
            "sem": float(sem),
            "sd_between": float(sd_between),
            "sd_within": float(sd_within),
            "total_sd": float(total_sd),
            "per_replica_means": values.tolist(),
            "per_replica_stds": stds.tolist(),
        }

    # Per-replica summary
    for r in replica_results:
        rep_summary = {"replica": r["replica"]}
        for comp in components:
            if comp in r["parsed"]["delta"]:
                rep_summary[comp] = r["parsed"]["delta"][comp]["mean"]
                rep_summary[f"{comp}_std"] = r["parsed"]["delta"][comp]["std"]
        ensemble["per_replica"].append(rep_summary)

    ensemble["frame_energies"] = all_frame_energies

    # Compute QHA entropy for each replica
    qha_results_list = []
    print(f"\n  Computing QHA entropy corrections...")
    for r in replica_results:
        rep = r["replica"]
        qha = compute_qha_entropy(complex_name, rep, analysis_window, atom_selection="ca")
        if qha:
            qha_results_list.append(qha)

    ensemble["qha_results"] = qha_results_list
    if qha_results_list:
        ensemble["qha_ensemble"] = aggregate_qha_results(qha_results_list)

        # Per-replica dG = dH + TdS, so that the SEM is taken across replicas and
        # captures the dH-TdS correlation (the same replica supplies both terms).
        # Propagating SEM(dH) and SEM(TdS) independently ignores it and overestimates
        # the uncertainty.
        dH_per_rep = {}
        for r in replica_results:
            if "TOTAL" in r["parsed"]["delta"]:
                dH_per_rep[r["replica"]] = r["parsed"]["delta"]["TOTAL"]["mean"]

        dG_qh_list = []
        for qha_r in qha_results_list:
            rep = qha_r["replica"]
            if rep in dH_per_rep:
                dG_qh_list.append(dH_per_rep[rep] + qha_r["quasi_harmonic"]["TdS"])

        if dG_qh_list:
            dG_qh_arr = np.array(dG_qh_list)
            n_qha = len(dG_qh_arr)
            target = ensemble["qha_ensemble"]["quasi_harmonic"]
            target["dG_mean"] = float(np.mean(dG_qh_arr))
            target["dG_std"] = float(np.std(dG_qh_arr, ddof=1)) if n_qha > 1 else 0.0
            target["dG_sem"] = (float(np.std(dG_qh_arr, ddof=1) / np.sqrt(n_qha))
                                if n_qha > 1 else 0.0)
            target["per_replica_dG"] = dG_qh_arr.tolist()

    return ensemble


# =============================================================================
# CONVERGENCE ASSESSMENT
# =============================================================================

def convergence_cumulative_mean(frame_energies):
    """
    Compute cumulative mean of per-frame delta TOTAL for each replica.

    Returns:
        dict: {replica: {"frames": np.array, "cumulative_mean": np.array}}
    """
    results = {}
    for rep, data in frame_energies.items():
        total_key = "delta_TOTAL"
        if total_key not in data:
            continue
        values = data[total_key]
        cumsum = np.cumsum(values)
        cumulative_mean = cumsum / np.arange(1, len(values) + 1)
        results[rep] = {
            "frames": np.arange(1, len(values) + 1),
            "cumulative_mean": cumulative_mean,
            "final_mean": float(cumulative_mean[-1]),
        }
    return results


def convergence_block_averaging(frame_energies, n_blocks=5):
    """
    Block averaging: divide frames into blocks and compute per-block means.

    Returns:
        dict: {replica: {"block_means": list, "block_stds": list, "overall_mean": float}}
    """
    results = {}
    for rep, data in frame_energies.items():
        total_key = "delta_TOTAL"
        if total_key not in data:
            continue
        values = data[total_key]
        block_size = len(values) // n_blocks
        if block_size < 1:
            continue

        block_means = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size if i < n_blocks - 1 else len(values)
            block_means.append(float(np.mean(values[start:end])))

        results[rep] = {
            "n_blocks": n_blocks,
            "block_size": block_size,
            "block_means": block_means,
            "block_spread": float(np.std(block_means, ddof=1)),
            "overall_mean": float(np.mean(values)),
        }
    return results


def convergence_inter_replica(ensemble):
    """
    Assess inter-replica consistency.

    Returns:
        dict with outlier detection results
    """
    if "TOTAL" not in ensemble["components"]:
        return {}

    means = np.array(ensemble["components"]["TOTAL"]["per_replica_means"])
    grand_mean = ensemble["components"]["TOTAL"]["grand_mean"]
    sd = ensemble["components"]["TOTAL"]["sd_between"]

    results = {
        "grand_mean": float(grand_mean),
        "sd_between": float(sd),
        "replicas": [],
    }

    for i, m in enumerate(means):
        rep_num = ensemble["per_replica"][i]["replica"]
        deviation = abs(m - grand_mean) / sd if sd > 0 else 0
        is_outlier = deviation > 2.0
        results["replicas"].append({
            "replica": rep_num,
            "mean": float(m),
            "deviation_sigma": float(deviation),
            "is_outlier": bool(is_outlier),
        })

    return results


def convergence_autocorrelation(frame_energies, max_lag=None):
    """
    Compute autocorrelation function and estimate correlation time.

    Returns:
        dict: {replica: {"acf": np.array, "correlation_time": float, "effective_n": float}}
    """
    results = {}
    for rep, data in frame_energies.items():
        total_key = "delta_TOTAL"
        if total_key not in data:
            continue
        values = data[total_key]
        n = len(values)
        if n < 10:
            continue

        if max_lag is None:
            max_lag_val = min(n // 2, 200)
        else:
            max_lag_val = min(max_lag, n // 2)

        # Compute normalized ACF
        mean = np.mean(values)
        var = np.var(values)
        if var == 0:
            continue

        acf = np.zeros(max_lag_val)
        centered = values - mean
        for lag in range(max_lag_val):
            acf[lag] = np.mean(centered[:n - lag] * centered[lag:]) / var

        # Estimate correlation time using Sokal's automatic windowing.
        # Integrate ACF until lag M >= C * tau (C=5, standard choice).
        # This is more robust than truncating at the first negative value,
        # which can underestimate tau for oscillatory ACFs.
        # Ref: Sokal, "Monte Carlo Methods in Statistical Mechanics" (1997).
        C_SOKAL = 5.0
        tau = 0.5  # Start with 0.5 for the lag=0 term (ACF(0)/2)
        sokal_converged = False
        for lag in range(1, max_lag_val):
            tau += acf[lag]
            # Stop when the window is wide enough relative to current tau
            if lag >= C_SOKAL * tau:
                sokal_converged = True
                break

        if not sokal_converged:
            print(f"    WARNING: Sokal windowing for rep_{rep} did not converge "
                  f"within max_lag={max_lag_val}. tau={tau:.1f} may be "
                  f"underestimated (strong long-range correlations).")

        effective_n = n / (2 * tau) if tau > 0 else n

        results[rep] = {
            "acf": acf[:min(50, max_lag_val)].tolist(),  # Store first 50 lags
            "correlation_time": float(tau),
            "effective_n": float(effective_n),
            "n_frames": n,
            "sokal_converged": sokal_converged,
        }
    return results


# =============================================================================
# VALIDATION
# =============================================================================

def run_validation(ensemble, analysis_window):
    """
    Run the validation checklist on the ensemble results.

    Returns:
        dict with validation results
    """
    validations = {
        "all_pass": True,
        "checks": [],
    }

    delta = ensemble["components"]

    # 1. Bonded term cancellation (verbose=2 should show these as 0)
    bonded_terms = ["BOND", "ANGLE", "DIHED", "UB", "IMP", "CMAP"]
    for term in bonded_terms:
        if term in delta:
            val = delta[term]["grand_mean"]
            is_zero = abs(val) < 0.01
            validations["checks"].append({
                "name": f"Bonded term cancellation ({term})",
                "value": val,
                "pass": is_zero,
                "expected": "0.0 (single-trajectory protocol)",
                "detail": f"{term} delta = {val:.4f} kcal/mol",
            })
            if not is_zero:
                validations["all_pass"] = False

    # 2. Sign of dG_total
    if "TOTAL" in delta:
        total_mean = delta["TOTAL"]["grand_mean"]
        is_negative = total_mean < 0
        validations["checks"].append({
            "name": "Sign of dG_total",
            "value": total_mean,
            "pass": is_negative,
            "expected": "Negative (favorable binding)",
            "detail": f"dG_total = {total_mean:.2f} kcal/mol",
        })
        if not is_negative:
            validations["all_pass"] = False

    # 3. Electrostatic balance
    if "EEL" in delta and "EPB" in delta:
        eel = delta["EEL"]["grand_mean"]
        epb = delta["EPB"]["grand_mean"]
        net_elec = eel + epb

        eel_negative = eel < 0
        epb_positive = epb > 0

        validations["checks"].append({
            "name": "Electrostatic balance (EEL)",
            "value": eel,
            "pass": eel_negative,
            "expected": "Large negative",
            "detail": f"EEL = {eel:.2f} kcal/mol",
        })
        validations["checks"].append({
            "name": "Electrostatic balance (EPB)",
            "value": epb,
            "pass": epb_positive,
            "expected": "Large positive (desolvation penalty)",
            "detail": f"EPB = {epb:.2f} kcal/mol",
        })
        validations["checks"].append({
            "name": "Net electrostatic contribution",
            "value": net_elec,
            "pass": True,  # Informational
            "expected": "Modest relative to individual terms",
            "detail": f"EEL + EPB = {net_elec:.2f} kcal/mol "
                       f"(|net/EEL| = {abs(net_elec/eel):.2f})" if eel != 0 else "N/A",
        })
        if not eel_negative or not epb_positive:
            validations["all_pass"] = False

    # 4. Magnitude check
    if "VDWAALS" in delta:
        vdw = delta["VDWAALS"]["grand_mean"]
        in_range = -100 <= vdw <= -25
        validations["checks"].append({
            "name": "VDW magnitude check",
            "value": vdw,
            "pass": in_range,
            "expected": "Between -100 and -25 kcal/mol for peptide-HLA (8-11 mer)",
            "detail": f"VDWAALS = {vdw:.2f} kcal/mol",
        })
        if not in_range:
            # Warning but don't fail
            pass

    if "TOTAL" in delta:
        total_mean = delta["TOTAL"]["grand_mean"]
        in_range = -100 <= total_mean <= -10
        validations["checks"].append({
            "name": "Total dG magnitude check",
            "value": total_mean,
            "pass": in_range,
            "expected": "Between -100 and -10 kcal/mol for peptide-HLA",
            "detail": f"TOTAL = {total_mean:.2f} kcal/mol",
        })

    # 5. SD vs mean
    if "TOTAL" in delta:
        total_mean = delta["TOTAL"]["grand_mean"]
        total_sd = delta["TOTAL"]["sd_between"]
        sd_ok = total_sd < abs(total_mean)
        validations["checks"].append({
            "name": "SD vs mean",
            "value": total_sd,
            "pass": sd_ok,
            "expected": "SD < |mean| (adequate sampling)",
            "detail": f"SD = {total_sd:.2f}, |mean| = {abs(total_mean):.2f}",
        })
        if not sd_ok:
            validations["all_pass"] = False

    # 6. Inter-replica outlier detection
    if "TOTAL" in delta:
        means = delta["TOTAL"]["per_replica_means"]
        grand_mean = delta["TOTAL"]["grand_mean"]
        sd = delta["TOTAL"]["sd_between"]
        outliers = []
        for i, m in enumerate(means):
            # Use actual replica number from per_replica data
            rep_num = ensemble["per_replica"][i]["replica"] if i < len(ensemble["per_replica"]) else i + 1
            if sd > 0:
                dev = abs(m - grand_mean) / sd
                if dev > 2.0:
                    outliers.append(f"rep_{rep_num} ({m:.2f}, {dev:.1f} sigma)")
        validations["checks"].append({
            "name": "Inter-replica outliers (>2 sigma)",
            "value": len(outliers),
            "pass": len(outliers) == 0,
            "expected": "No outliers",
            "detail": f"Outliers: {', '.join(outliers)}" if outliers else "None",
        })

    return validations


# =============================================================================
# CONVERGENCE ACROSS ANALYSIS WINDOWS
# =============================================================================

# Human-readable labels for each convergence dG method (see config.CONVERGENCE_DG_METHOD)
DG_METHOD_LABELS = {
    "dh": r"$\Delta H_{\rm MM\text{-}PBSA}$",
    "qh": r"$\Delta G_{\rm bind}$ (QH)",
}


def _extract_window_dg(ens, method):
    """
    Extract the binding free energy for one analysis window's ensemble dict,
    for the entropy-correction method named by `method`.

    Pulls the entropy-corrected dG (or the bare MM-PBSA enthalpy) from the
    locations where Phase 4 stores it, so the convergence figure can track the
    real free energy rather than the enthalpy alone.

    Args:
        ens: ensemble dict for a single window (the same structure saved to
             ensemble_summary_<window>.json), expected to contain "components",
             "per_replica", and, for the entropy-corrected method,
             "qha_ensemble".
        method: one of the keys documented in config.CONVERGENCE_DG_METHOD
                ("dh" or "qh").

    Returns:
        (mean, sem, per_replica) where
            mean: float or None  -- ensemble grand-mean dG (kcal/mol)
            sem:  float or None  -- SEM across replicas of that dG
            per_replica: dict {replica_number: dG_value} (may be empty)
        Missing data degrades gracefully to None / {} so the caller can fall
        back without raising.
    """
    # Replica numbers in the order the per-replica arrays were built.
    rep_numbers = [r.get("replica") for r in ens.get("per_replica", [])]

    def _map_per_replica(values):
        """Zip an ordered per-replica array onto its replica numbers."""
        if not values or len(values) != len(rep_numbers):
            return {}
        return {rep: val for rep, val in zip(rep_numbers, values)}

    if method == "dh":
        total = ens.get("components", {}).get("TOTAL", {})
        per_rep = {r["replica"]: r.get("TOTAL") for r in ens.get("per_replica", [])}
        return total.get("grand_mean"), total.get("sem"), per_rep

    if method == "qh":
        sub = (ens.get("qha_ensemble") or {}).get("quasi_harmonic") or {}
        return (sub.get("dG_mean"), sub.get("dG_sem"),
                _map_per_replica(sub.get("per_replica_dG")))

    raise ValueError(f"Unknown convergence dG method: {method!r}")


def convergence_across_windows(all_ensembles):
    """
    Analyze how binding free energy converges across multiple analysis windows.

    Computes dG (TOTAL grand_mean +/- SEM) at each window, paired differences
    vs the reference window (500ns), per-replica trajectories across windows,
    and an overall convergence assessment.

    Args:
        all_ensembles: dict mapping window name -> ensemble dict
                       (e.g., {"100ns": {...}, "200ns": {...}, ...}).
                       Keys must be a subset of config.ANALYSIS_WINDOWS.

    Returns:
        dict with keys:
            - windows: list of window names in order
            - convergence_summary: {window: {mean, sem, n_replicas}}  (MM-PBSA dH)
            - per_replica_across_windows: {rep: {window: dG_total}}    (MM-PBSA dH)
            - pairwise_vs_reference: {window: {difference, sem_diff, significant}}
            - convergence_assessment: str (human-readable assessment)
            - dg_method: str (config.CONVERGENCE_DG_METHOD, e.g. "qh")
            - dg_method_label: str (display label for the plotted dG)
            - dg_convergence_summary: {window: {mean, sem, n_replicas}}  (chosen dG)
            - per_replica_dg_across_windows: {rep: {window: dG}}          (chosen dG)

        The dh-based keys (convergence_summary / per_replica_across_windows /
        pairwise_vs_reference / convergence_assessment) are kept unchanged for
        backward compatibility; the dg_* keys hold the entropy-corrected binding
        free energy that the convergence figure now plots.
    """
    # Order windows by simulation time (ascending)
    ordered_windows = [w for w in config.ALL_WINDOW_NAMES if w in all_ensembles]

    dg_method = config.CONVERGENCE_DG_METHOD
    result = {
        "windows": ordered_windows,
        "convergence_summary": {},
        "per_replica_across_windows": {},
        "pairwise_vs_reference": {},
        "convergence_assessment": "",
        "dg_method": dg_method,
        "dg_method_label": DG_METHOD_LABELS.get(dg_method, dg_method),
        "dg_convergence_summary": {},
        "per_replica_dg_across_windows": {},
    }

    # 1. Extract per-window ensemble dG (TOTAL)
    for window in ordered_windows:
        ens = all_ensembles[window]
        total = ens.get("components", {}).get("TOTAL", {})
        result["convergence_summary"][window] = {
            "mean": total.get("grand_mean"),
            "sem": total.get("sem"),
            "n_replicas": ens.get("n_replicas", 0),
        }

    # 2. Build per-replica dG across windows
    # Map: {replica_num: {window: dG_total}}
    per_rep_maps = {}
    for window in ordered_windows:
        ens = all_ensembles[window]
        for r in ens.get("per_replica", []):
            rep = r["replica"]
            if rep not in per_rep_maps:
                per_rep_maps[rep] = {}
            per_rep_maps[rep][window] = r.get("TOTAL")
    result["per_replica_across_windows"] = per_rep_maps

    # 2b. Entropy-corrected dG across windows (this is what the figure plots).
    # Falls back to the MM-PBSA enthalpy ("dh") if the chosen method's data is
    # missing for a window, so legacy ensembles still produce a figure.
    dg_per_rep_maps = {}
    for window in ordered_windows:
        ens = all_ensembles[window]
        mean, sem, per_rep = _extract_window_dg(ens, dg_method)
        if mean is None:
            mean, sem, per_rep = _extract_window_dg(ens, "dh")
        result["dg_convergence_summary"][window] = {
            "mean": mean,
            "sem": sem,
            "n_replicas": ens.get("n_replicas", 0),
        }
        for rep, val in per_rep.items():
            dg_per_rep_maps.setdefault(rep, {})[window] = val
    result["per_replica_dg_across_windows"] = dg_per_rep_maps

    # 3. Pairwise comparison vs reference window
    ref_window = config.REFERENCE_WINDOW
    if ref_window in all_ensembles:
        ref_ens = all_ensembles[ref_window]
        ref_per_rep = {r["replica"]: r.get("TOTAL") for r in ref_ens.get("per_replica", [])}

        for window in ordered_windows:
            if window == ref_window:
                continue
            ens = all_ensembles[window]
            win_per_rep = {r["replica"]: r.get("TOTAL") for r in ens.get("per_replica", [])}

            # Paired differences (correct for correlation since shorter windows
            # are subsets of longer ones)
            common_reps = sorted(set(ref_per_rep.keys()) & set(win_per_rep.keys()))
            paired_diffs = []
            for rep in common_reps:
                v_ref = ref_per_rep[rep]
                v_win = win_per_rep[rep]
                if v_ref is not None and v_win is not None:
                    paired_diffs.append(v_win - v_ref)

            if len(paired_diffs) > 1:
                paired_diffs = np.array(paired_diffs)
                mean_diff = float(np.mean(paired_diffs))
                sem_diff = float(np.std(paired_diffs, ddof=1) / np.sqrt(len(paired_diffs)))
            elif len(paired_diffs) == 1:
                mean_diff = float(paired_diffs[0])
                sem_diff = None
            else:
                mean_diff = None
                sem_diff = None

            result["pairwise_vs_reference"][window] = {
                "reference": ref_window,
                "difference": mean_diff,
                "sem_diff": sem_diff,
                "n_pairs": len(paired_diffs),
                "significant": bool(
                    mean_diff is not None and sem_diff is not None
                    and sem_diff > 0 and abs(mean_diff) > 2 * sem_diff
                ),
            }

    # 4. Convergence assessment
    result["convergence_assessment"] = _assess_convergence(result)

    return result


def _assess_convergence(conv_result):
    """
    Generate a human-readable convergence assessment.

    Checks whether the last 3 windows have dG values within 2 SEM of each other
    and reports the overall trend.

    Args:
        conv_result: dict from convergence_across_windows()

    Returns:
        str: Assessment text
    """
    windows = conv_result["windows"]
    summary = conv_result["convergence_summary"]

    if len(windows) < 2:
        return "Insufficient windows for convergence assessment."

    # Check if the last N windows are mutually consistent (within 2 SEM)
    # Use the last 3 windows if available, otherwise all
    check_windows = windows[-3:] if len(windows) >= 3 else windows
    check_means = []
    check_sems = []
    for w in check_windows:
        s = summary.get(w, {})
        if s.get("mean") is not None and s.get("sem") is not None:
            check_means.append(s["mean"])
            check_sems.append(s["sem"])

    if len(check_means) < 2:
        return "Insufficient data for convergence assessment."

    # Maximum pairwise difference among the check windows
    max_diff = max(check_means) - min(check_means)
    avg_sem = np.mean(check_sems)

    # Also check pairwise significance vs reference
    pairwise = conv_result.get("pairwise_vs_reference", {})
    significant_diffs = [w for w, p in pairwise.items() if p.get("significant")]

    lines = []
    if max_diff <= 2 * avg_sem:
        lines.append(
            f"CONVERGED: The last {len(check_windows)} windows "
            f"({', '.join(check_windows)}) have dG values within 2 SEM "
            f"(max spread = {max_diff:.2f} kcal/mol, avg SEM = {avg_sem:.2f})."
        )
    else:
        lines.append(
            f"NOT CONVERGED: The last {len(check_windows)} windows "
            f"({', '.join(check_windows)}) show a spread of {max_diff:.2f} kcal/mol "
            f"(> 2 x avg SEM = {2*avg_sem:.2f}). Consider longer simulation times."
        )

    if significant_diffs:
        lines.append(
            f"Windows with significant difference vs {config.REFERENCE_WINDOW}: "
            f"{', '.join(significant_diffs)}."
        )
    else:
        lines.append(
            f"No window shows a significant difference vs {config.REFERENCE_WINDOW}."
        )

    return " ".join(lines)


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plots(complex_name, all_ensembles, convergence_data=None,
                   window_comparison=None):
    """
    Generate all analysis plots.

    Args:
        complex_name: Complex directory name
        all_ensembles: dict {window_name: ensemble_dict}
        convergence_data: dict {window_name: per-window convergence data}
        window_comparison: dict from convergence_across_windows()
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available. Skipping plot generation.")
        return

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.dpi": 150,
    })

    results_dir = utils.get_results_dir(complex_name)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for window in sorted(all_ensembles.keys()):
        ensemble = all_ensembles[window]

        # Plot 1: Per-replica dG bar chart
        _plot_replica_bars(ensemble, window, plots_dir)

        # Plot 2: Energy component decomposition
        _plot_energy_components(ensemble, window, plots_dir)

        # Convergence plots
        if convergence_data and window in convergence_data:
            conv = convergence_data[window]

            # Plot 3: Cumulative mean
            if "cumulative_mean" in conv:
                _plot_cumulative_mean(conv["cumulative_mean"], window, plots_dir)

            # Plot 4: Block averaging
            if "block_averaging" in conv:
                _plot_block_averaging(conv["block_averaging"], window, plots_dir)

            # Plot 5: Autocorrelation
            if "autocorrelation" in conv:
                _plot_autocorrelation(conv["autocorrelation"], window, plots_dir)

    # Plot 6: Convergence across windows
    if window_comparison and len(all_ensembles) >= 2:
        _plot_convergence_across_windows(window_comparison, plots_dir)

    print(f"  Plots saved to {plots_dir}")


def _plot_replica_bars(ensemble, window, plots_dir):
    import matplotlib.pyplot as plt

    if "TOTAL" not in ensemble["components"]:
        return

    comp = ensemble["components"]["TOTAL"]
    means = comp["per_replica_means"]
    grand_mean = comp["grand_mean"]
    sem = comp["sem"]
    n = len(means)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(1, n + 1), means, color="#4C72B0", alpha=0.8,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(grand_mean, color="#C44E52", ls="--", lw=2,
               label=f"Mean = {grand_mean:.1f} +/- {sem:.1f} kcal/mol")
    ax.fill_between([0.5, n + 0.5], grand_mean - sem, grand_mean + sem,
                    alpha=0.2, color="#C44E52")
    ax.set_xlabel("Replica")
    ax.set_ylabel(r"$\Delta G_{\rm bind}$ (kcal/mol)")
    ax.set_title(f"MM-PBSA binding free energy ({window})")
    ax.set_xticks(range(1, n + 1))
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"replica_dG_{window}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def _plot_energy_components(ensemble, window, plots_dir):
    import matplotlib.pyplot as plt

    components = ["VDWAALS", "EEL", "EPB", "ENPOLAR", "TOTAL"]
    available = [c for c in components if c in ensemble["components"]]
    if not available:
        return

    labels = available
    means = [ensemble["components"][c]["grand_mean"] for c in available]
    sems = [ensemble["components"][c]["sem"] for c in available]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][:len(available)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=sems, capsize=4, color=colors,
           edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Energy (kcal/mol)")
    ax.set_title(f"MM-PBSA energy decomposition ({window})")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"energy_components_{window}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def _plot_cumulative_mean(cum_data, window, plots_dir):
    import matplotlib.pyplot as plt

    if not cum_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for rep, data in sorted(cum_data.items()):
        ax.plot(data["frames"], data["cumulative_mean"],
                label=f"rep_{rep}", alpha=0.7)
    ax.set_xlabel("Frame number")
    ax.set_ylabel(r"Cumulative mean $\Delta G$ (kcal/mol)")
    ax.set_title(f"Convergence: cumulative mean ({window})")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"convergence_cumulative_{window}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def _plot_block_averaging(block_data, window, plots_dir):
    import matplotlib.pyplot as plt

    if not block_data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    reps = sorted(block_data.keys())
    n_blocks = block_data[reps[0]]["n_blocks"] if reps else 5
    x = np.arange(1, n_blocks + 1)
    width = 0.8 / len(reps)

    for i, rep in enumerate(reps):
        offset = (i - len(reps) / 2 + 0.5) * width
        ax.bar(x + offset, block_data[rep]["block_means"],
               width=width, alpha=0.7, label=f"rep_{rep}")

    ax.set_xlabel("Block number")
    ax.set_ylabel(r"$\Delta G$ (kcal/mol)")
    ax.set_title(f"Convergence: block averaging ({window})")
    ax.set_xticks(x)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"convergence_blocks_{window}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def _plot_autocorrelation(acf_data, window, plots_dir):
    import matplotlib.pyplot as plt

    if not acf_data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for rep, data in sorted(acf_data.items()):
        acf = data["acf"]
        lags = np.arange(len(acf))
        ax.plot(lags, acf, label=f"rep_{rep} (tau={data['correlation_time']:.1f})",
                alpha=0.7)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"Convergence: autocorrelation ({window})")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"convergence_acf_{window}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def _plot_convergence_across_windows(window_comparison, plots_dir):
    """
    Plot convergence of the binding free energy across analysis windows.

    The plotted quantity is the dG selected by config.CONVERGENCE_DG_METHOD
    ("qh" by default), read from the
    `dg_convergence_summary` / `per_replica_dg_across_windows` fields produced by
    convergence_across_windows(). If those are absent (legacy data) it falls back
    to the MM-PBSA enthalpy (`convergence_summary` / `per_replica_across_windows`).

    Two-panel figure:
      - Panel A: dG (mean +/- SEM) vs simulation time
      - Panel B: Per-replica dG trajectories across windows

    Args:
        window_comparison: dict from convergence_across_windows()
        plots_dir: Directory to save the plot
    """
    import matplotlib.pyplot as plt

    windows = window_comparison.get("windows", [])
    if len(windows) < 2:
        return

    # Prefer the entropy-corrected dG; fall back to MM-PBSA enthalpy (legacy).
    summary = window_comparison.get("dg_convergence_summary")
    per_rep = window_comparison.get("per_replica_dg_across_windows")
    y_label = window_comparison.get("dg_method_label", r"$\Delta G_{\rm bind}$")
    if not summary:
        summary = window_comparison.get("convergence_summary", {})
        per_rep = window_comparison.get("per_replica_across_windows", {})
        y_label = r"$\Delta H_{\rm MM\text{-}PBSA}$"
    per_rep = per_rep or {}
    y_label = f"{y_label} (kcal/mol)"

    # Extract simulation times (strip "ns" suffix) for x-axis
    x_times = []
    means = []
    sems = []
    for w in windows:
        x_times.append(int(w.replace("ns", "")))
        s = summary.get(w, {})
        means.append(s.get("mean", 0))
        sems.append(s.get("sem", 0) if s.get("sem") is not None else 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Ensemble mean +/- SEM vs simulation time
    ax = axes[0]
    ax.set_axisbelow(True)
    ax.grid(True, color="gray", linestyle="-", linewidth=0.6, alpha=0.3)
    ax.errorbar(x_times, means, yerr=sems, fmt="o-", color="#4C72B0",
                capsize=4, capthick=1.5, markersize=8, linewidth=2, zorder=3)
    ax.fill_between(x_times,
                    [m - s for m, s in zip(means, sems)],
                    [m + s for m, s in zip(means, sems)],
                    alpha=0.15, color="#4C72B0")
    ax.set_xlabel("Simulation time (ns)")
    ax.set_ylabel(y_label)
    ax.set_title("Convergence of binding free energy")
    ax.set_xticks(x_times)

    # Panel B: Per-replica trajectories
    ax = axes[1]
    ax.set_axisbelow(True)
    ax.grid(True, color="gray", linestyle="-", linewidth=0.6, alpha=0.3)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(per_rep), 1)))
    all_y = []  # collect every plotted value to size the reserved legend band
    for i, (rep, vals) in enumerate(sorted(per_rep.items())):
        rep_x = []
        rep_y = []
        for w in windows:
            if w in vals and vals[w] is not None:
                rep_x.append(int(w.replace("ns", "")))
                rep_y.append(vals[w])
        if rep_y:
            ax.plot(rep_x, rep_y, "o-", color=colors[i % len(colors)],
                    alpha=0.5, linewidth=1, markersize=4, label=f"rep_{rep}")
            all_y.extend(rep_y)

    # Overlay ensemble mean
    ax.plot(x_times, means, "s-", color="black", linewidth=2.5, markersize=7,
            zorder=10, label="Ensemble mean")
    all_y.extend([m for m in means if m is not None])
    ax.set_xlabel("Simulation time (ns)")
    ax.set_ylabel(y_label)
    ax.set_title("Per-replica convergence")
    ax.set_xticks(x_times)

    # Reserve empty headroom at the top of the panel and drop the legend into it.
    # This is complex-independent: the band is guaranteed clear of every curve
    # (we expand ylim above the data max) and the title sits outside the axes, so
    # the legend can overlap neither the data nor the title. ncol=6 keeps the
    # 11-entry legend to two short rows that fit inside the reserved band.
    if all_y:
        y_lo, y_hi = min(all_y), max(all_y)
        y_range = (y_hi - y_lo) or 1.0
        ax.set_ylim(y_lo - 0.08 * y_range, y_hi + 0.30 * y_range)
    ax.legend(fontsize=7, ncol=6, columnspacing=1.0, handletextpad=0.4,
              loc="upper right", borderaxespad=0.5,
              frameon=True, framealpha=0.9, edgecolor="0.7")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "convergence_across_windows.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_results(complex_name, analysis_window="all"):
    """
    Run complete analysis for a complex.

    Args:
        complex_name: Complex directory name
        analysis_window: A window name (e.g. "500ns"), or "all" for all windows

    Returns:
        dict with all analysis results
    """
    print(f"\n{'='*60}")
    print(f"Phase 4: Analysis for {complex_name}")
    print(f"{'='*60}")

    results_dir = utils.get_results_dir(complex_name)
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    convergence_data = {}

    # Determine windows
    if analysis_window == "all":
        windows = list(config.ANALYSIS_WINDOWS.keys())
    else:
        windows = [analysis_window]

    for window in windows:
        print(f"\n--- Processing {window} analysis ---")

        # Aggregate results
        ensemble = aggregate_replicas(complex_name, window)
        if ensemble is None:
            print(f"  Skipping {window}: no results")
            continue

        all_results[window] = ensemble

        # Print ensemble summary
        print(f"\n  ENSEMBLE RESULTS ({window}, {ensemble['n_replicas']} replicas):")
        for comp in ["VDWAALS", "EEL", "EPB", "ENPOLAR", "TOTAL"]:
            if comp in ensemble["components"]:
                c = ensemble["components"][comp]
                print(f"    {comp:<12} {c['grand_mean']:>10.2f} +/- {c['sem']:>6.2f} kcal/mol "
                      f"(SD_between={c['sd_between']:.2f})")

        # QHA ensemble summary
        if ensemble.get("qha_ensemble"):
            qha_ens = ensemble["qha_ensemble"]
            print(f"\n  QUASI-HARMONIC ENTROPY ({window}, {qha_ens['n_replicas']} replicas, {qha_ens['atom_selection']} atoms):")
            qh = qha_ens["quasi_harmonic"]
            print(f"    Quasi-Harm -TΔS = {qh['TdS_mean']:>10.2f} +/- {qh['TdS_sem']:>6.2f} kcal/mol")
            if "dG_mean" in qh:
                print(f"    Quasi-Harm  dG  = {qh['dG_mean']:>10.2f} +/- {qh['dG_sem']:>6.2f} kcal/mol")
            print(f"    Per-replica Quasi-Harm: {[f'{v:.2f}' for v in qh['per_replica_TdS']]}")

        # Convergence assessment
        if ensemble["frame_energies"]:
            print(f"\n  Convergence assessment ({window}):")
            conv = {}

            # 1. Cumulative mean
            conv["cumulative_mean"] = convergence_cumulative_mean(ensemble["frame_energies"])
            if conv["cumulative_mean"]:
                print(f"    Cumulative mean: computed for {len(conv['cumulative_mean'])} replicas")

            # 2. Block averaging
            conv["block_averaging"] = convergence_block_averaging(ensemble["frame_energies"])
            if conv["block_averaging"]:
                for rep, ba in sorted(conv["block_averaging"].items()):
                    print(f"    Block averaging rep_{rep}: spread = {ba['block_spread']:.2f} kcal/mol")

            # 3. Inter-replica consistency
            conv["inter_replica"] = convergence_inter_replica(ensemble)
            if conv["inter_replica"].get("replicas"):
                outliers = [r for r in conv["inter_replica"]["replicas"] if r["is_outlier"]]
                if outliers:
                    print(f"    Inter-replica outliers: {[r['replica'] for r in outliers]}")
                else:
                    print(f"    Inter-replica consistency: no outliers detected")

            # 4. Autocorrelation
            conv["autocorrelation"] = convergence_autocorrelation(ensemble["frame_energies"])
            if conv["autocorrelation"]:
                for rep, ac in sorted(conv["autocorrelation"].items()):
                    print(f"    Autocorrelation rep_{rep}: tau = {ac['correlation_time']:.1f} frames, "
                          f"N_eff = {ac['effective_n']:.0f}/{ac['n_frames']}")

                # Compute autocorrelation-adjusted within-replica statistics (C1 fix).
                # The ensemble SEM (sd_between / sqrt(n)) is correct as-is since it
                # uses inter-replica variance.  The within-replica SD is affected by
                # frame autocorrelation: effective SD_within = SD_naive * sqrt(2*tau).
                taus = [ac["correlation_time"] for ac in conv["autocorrelation"].values()]
                median_tau = float(np.median(taus))
                median_neff_ratio = float(np.median([
                    ac["effective_n"] / ac["n_frames"]
                    for ac in conv["autocorrelation"].values()
                ]))

                if "TOTAL" in ensemble["components"]:
                    naive_sd_within = ensemble["components"]["TOTAL"]["sd_within"]
                    adjusted_sd_within = naive_sd_within * np.sqrt(2 * median_tau)
                    adjusted_total_sd = float(np.sqrt(
                        ensemble["components"]["TOTAL"]["sd_between"]**2
                        + adjusted_sd_within**2
                    ))
                    ensemble["components"]["TOTAL"]["adjusted_sd_within"] = float(adjusted_sd_within)
                    ensemble["components"]["TOTAL"]["adjusted_total_sd"] = adjusted_total_sd
                    ensemble["components"]["TOTAL"]["median_tau"] = median_tau
                    ensemble["components"]["TOTAL"]["median_neff_ratio"] = median_neff_ratio

                if median_neff_ratio < 0.5:
                    print(f"    WARNING: Strong autocorrelation detected "
                          f"(median N_eff/N = {median_neff_ratio:.2f}, "
                          f"median tau = {median_tau:.1f}). "
                          f"Within-replica error bars may be underestimated.")

            convergence_data[window] = conv
        else:
            print(f"  No per-frame energies available for convergence analysis")

        # Validation
        print(f"\n  Validation checklist ({window}):")
        validation = run_validation(ensemble, window)
        all_results[f"{window}_validation"] = validation
        for check in validation["checks"]:
            status = "PASS" if check["pass"] else "FAIL"
            print(f"    [{status}] {check['name']}: {check['detail']}")

        # Save ensemble summary
        summary_path = os.path.join(results_dir, f"ensemble_summary_{window}.json")
        # Convert numpy arrays to lists for JSON serialization
        save_data = {k: v for k, v in ensemble.items() if k != "frame_energies"}
        utils.save_metadata(summary_path, save_data)

    # Convergence across windows
    window_comparison = None
    ensembles_for_comparison = {w: all_results[w] for w in all_results
                                if w in config.ANALYSIS_WINDOWS}
    if len(ensembles_for_comparison) >= 2:
        print(f"\n--- Convergence Across Windows ---")
        window_comparison = convergence_across_windows(ensembles_for_comparison)
        all_results["window_comparison"] = window_comparison

        for w in window_comparison["windows"]:
            s = window_comparison["convergence_summary"][w]
            sem_str = f"+/- {s['sem']:.2f}" if s['sem'] is not None else "(no SEM)"
            print(f"  {w}: dG = {s['mean']:.2f} {sem_str} (n={s['n_replicas']})")

        for w, pw in window_comparison.get("pairwise_vs_reference", {}).items():
            if pw["difference"] is not None:
                sem_str = f"+/- {pw['sem_diff']:.2f}" if pw['sem_diff'] is not None else ""
                sig_str = " *" if pw["significant"] else ""
                print(f"  {w} vs {config.REFERENCE_WINDOW}: diff = {pw['difference']:.2f} "
                      f"{sem_str}{sig_str}")

        print(f"  Assessment: {window_comparison['convergence_assessment']}")

        # Save convergence across windows
        comp_path = os.path.join(results_dir, "convergence_across_windows.json")
        utils.save_metadata(comp_path, window_comparison)

        # Clean up old pairwise comparison file if it exists
        old_comp_path = os.path.join(results_dir, "comparison_500ns_vs_100ns.json")
        if os.path.exists(old_comp_path):
            os.remove(old_comp_path)
            print(f"  Removed stale {os.path.basename(old_comp_path)}")

    # Generate plots
    print(f"\n--- Generating plots ---")
    generate_plots(
        complex_name,
        ensembles_for_comparison,
        convergence_data,
        window_comparison,
    )

    # Save convergence data (without numpy arrays)
    for window, conv in convergence_data.items():
        conv_save = {}
        for key, val in conv.items():
            if key == "cumulative_mean":
                conv_save[key] = {
                    str(rep): {"final_mean": d["final_mean"],
                               "n_frames": len(d["frames"])}
                    for rep, d in val.items()
                }
            elif key == "autocorrelation":
                conv_save[key] = val
            elif key == "block_averaging":
                conv_save[key] = val
            elif key == "inter_replica":
                conv_save[key] = val
        conv_path = os.path.join(results_dir, f"convergence_{window}.json")
        utils.save_metadata(conv_path, conv_save)

    print(f"\n  Analysis complete! Results saved to {results_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Analyze MMPBSA results")
    parser.add_argument("complex_name", help="Complex directory name")
    parser.add_argument("--window", choices=config.ALL_WINDOW_NAMES + ["all"], default="all",
                        help="Analysis window (default: all)")
    args = parser.parse_args()
    analyze_results(args.complex_name, args.window)


if __name__ == "__main__":
    main()
