#!/usr/bin/env python3
"""Confirmatory secondary econometric re-analysis of Bastani et al. (2025, PNAS) microdata (K5).

Replicates `main_analysis.R` on the audited read-only CSVs (`final_data.csv` and
`gpt_answers_full.csv`) with exact class-clustered robust SEs AND 10,000-draw Rademacher
wild cluster bootstrap 95% CIs, plus pre-treatment `gpa_prev` balance check, full-sample
(including Honors), non-complier exclusion, and no-`gpa_prev` sensitivity specifications.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BASTANI_DATA_DIR", str(REPO_ROOT / "data" / "external" / "bastani_2025_pnas" / "main_regressions")))
OUTPUT_ROOT = Path(os.environ.get("RELEASE_OUTPUT_ROOT", str(REPO_ROOT)))
OUT_DIR = OUTPUT_ROOT / "results"


def wild_cluster_bootstrap_ci(
    df: pd.DataFrame,
    formula: str,
    cluster_col: str,
    target_vars: list[str],
    n_boot: int = 10000,
    seed: int = 20261119,
) -> dict:
    """Estimate OLS with cluster-robust SEs and Rademacher wild cluster bootstrap 95% CIs."""
    mod = smf.ols(formula, data=df)
    res_base = mod.fit()
    used_idx = res_base.model.data.row_labels
    clusters = df.loc[used_idx, cluster_col].astype(str).to_numpy()
    res_cl = res_base.get_robustcov_results(cov_type="cluster", groups=clusters)

    X = mod.exog
    y = mod.endog
    names = list(mod.exog_names)
    beta_hat = np.asarray(res_cl.params, dtype=float)
    se_cl = np.asarray(res_cl.bse, dtype=float)
    p_cl = np.asarray(res_cl.pvalues, dtype=float)
    resid = np.asarray(res_base.resid, dtype=float)
    yhat = np.asarray(res_base.fittedvalues, dtype=float)

    # Precompute (X'X)^{-1} X' for fast vectorized wild cluster bootstrap across 10,000 draws
    pinv_X = np.linalg.pinv(X)
    uniq_cl, inv_cl = np.unique(clusters, return_inverse=True)
    G = len(uniq_cl)

    # Build cluster residual matrix M of shape (p, G): column g = pinv_X[:, cl==g] @ resid[cl==g]
    M = np.zeros((X.shape[1], G), dtype=float)
    for g_idx in range(G):
        mask = inv_cl == g_idx
        M[:, g_idx] = pinv_X[:, mask] @ resid[mask]

    rng = np.random.default_rng(seed)
    # Draw Rademacher weights (+1 / -1) of shape (G, n_boot)
    W = rng.choice(np.array([-1.0, 1.0]), size=(G, n_boot))

    out = {"N": int(len(y)), "G_clusters": int(G), "r2": float(res_base.rsquared), "vars": {}}
    for tv in target_vars:
        idx = names.index(tv)
        b_pt = float(beta_hat[idx])
        se_pt = float(se_cl[idx])
        p_pt = float(p_cl[idx])
        # Bootstrap distribution of beta_hat_tv^* = b_pt + M[idx, :] @ W
        boot_deltas = M[idx, :] @ W  # (n_boot,)
        boot_betas = b_pt + boot_deltas
        ci_lo = float(np.quantile(boot_betas, 0.025))
        ci_hi = float(np.quantile(boot_betas, 0.975))
        p_boot = float(np.mean(np.abs(boot_deltas) >= abs(b_pt)))
        out["vars"][tv] = {
            "coef": b_pt,
            "se_cluster": se_pt,
            "p_cluster": p_pt,
            "ci95_cluster": [b_pt - 1.96 * se_pt, b_pt + 1.96 * se_pt],
            "ci95_wild_boot": [ci_lo, ci_hi],
            "p_wild_boot": p_boot,
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = DATA_DIR / "final_data.csv"
    gp = DATA_DIR / "gpt_answers_full.csv"
    sha_final = hashlib.sha256(fp.read_bytes()).hexdigest()
    sha_gpt = hashlib.sha256(gp.read_bytes()).hexdigest()

    df_full = pd.read_csv(fp)
    df_nonhon = df_full[df_full["Honors"] == 0].copy()

    # Exact Bastani et al. (2025) primary ITT specification (main_analysis.R lines 27 to 33)
    f_p2 = "Part2Tot ~ GPTBase + GPTTutor + gpa_prev + C(teacher) + C(Session) + C(Grader) + C(Year)"
    f_p3 = "Part3Tot ~ GPTBase + GPTTutor + gpa_prev + C(teacher) + C(Session) + C(Grader) + C(Year)"
    f_bal = "gpa_prev ~ GPTBase + GPTTutor + C(teacher) + C(Year)"
    f_p3_nogpa = "Part3Tot ~ GPTBase + GPTTutor + C(teacher) + C(Session) + C(Grader) + C(Year)"
    # Pre-registered Critic 6120aba1 robustness specs:
    # (a) Minimal stratum FE (Year + Session only, excluding single-arm teacher/Grader FEs)
    f_p2_minfe = "Part2Tot ~ GPTBase + GPTTutor + gpa_prev + C(Session) + C(Year)"
    f_p3_minfe = "Part3Tot ~ GPTBase + GPTTutor + gpa_prev + C(Session) + C(Year)"
    # (b) Session-1-only (eliminates multi-session attendance/attrition differences)
    f_p3_s1 = "Part3Tot ~ GPTBase + GPTTutor + gpa_prev + C(Year)"
    # (c) Student-collapsed mean across attended sessions (equal student weight)
    f_p3_coll = "Part3Tot ~ GPTBase + GPTTutor + gpa_prev + C(Year)"

    # Non-complier drop mask from main_analysis.R lines 41 to 45
    nc_mask = (
        ((df_nonhon["Class"] == "11S") & (df_nonhon["Session"] == 2))
        | ((df_nonhon["Class"] == "11S") & (df_nonhon["Session"] == 4))
        | ((df_nonhon["Class"] == "10A") & (df_nonhon["Session"] == 2))
        | ((df_nonhon["Class"] == "9J") & (df_nonhon["Session"] == 1))
        | ((df_nonhon["Class"] == "10B") & (df_nonhon["Session"] == 3))
    )
    df_comp = df_nonhon[~nc_mask].copy()
    df_s1 = df_nonhon[df_nonhon["Session"] == 1].copy()
    df_coll = (
        df_nonhon.groupby("Student ID", as_index=False)
        .agg({
            "Part2Tot": "mean",
            "Part3Tot": "mean",
            "gpa_prev": "first",
            "GPTBase": "first",
            "GPTTutor": "first",
            "Class": "first",
            "Year": "first",
        })
    )

    # Student-level unique table for pre-treatment gpa_prev balance check
    df_stud = df_nonhon.drop_duplicates(subset=["Student ID"]).copy()

    # Control means on complete-case primary estimation sample
    cc = df_nonhon.dropna(subset=["Part2Tot", "Part3Tot", "gpa_prev", "Grader", "teacher", "Session", "Year"])
    ctrl_cc = cc[cc["Treatment arm"] == "control"]
    ctrl_p2_mean = float(ctrl_cc["Part2Tot"].mean())
    ctrl_p3_mean = float(ctrl_cc["Part3Tot"].mean())
    ctrl_p3_sd = float(ctrl_cc["Part3Tot"].std(ddof=1))

    r_p2 = wild_cluster_bootstrap_ci(df_nonhon, f_p2, "Class", ["GPTBase", "GPTTutor"])
    r_p3 = wild_cluster_bootstrap_ci(df_nonhon, f_p3, "Class", ["GPTBase", "GPTTutor"])
    r_bal = wild_cluster_bootstrap_ci(df_stud, f_bal, "Class", ["GPTBase", "GPTTutor"])
    r_p3_comp = wild_cluster_bootstrap_ci(df_comp, f_p3, "Class", ["GPTBase", "GPTTutor"])
    r_p3_full = wild_cluster_bootstrap_ci(df_full, f_p3, "Class", ["GPTBase", "GPTTutor"])
    r_p3_nogpa = wild_cluster_bootstrap_ci(df_nonhon, f_p3_nogpa, "Class", ["GPTBase", "GPTTutor"])
    r_p2_minfe = wild_cluster_bootstrap_ci(df_nonhon, f_p2_minfe, "Class", ["GPTBase", "GPTTutor"])
    r_p3_minfe = wild_cluster_bootstrap_ci(df_nonhon, f_p3_minfe, "Class", ["GPTBase", "GPTTutor"])
    r_p3_s1 = wild_cluster_bootstrap_ci(df_s1, f_p3_s1, "Class", ["GPTBase", "GPTTutor"])
    r_p3_coll = wild_cluster_bootstrap_ci(df_coll, f_p3_coll, "Class", ["GPTBase", "GPTTutor"])

    # GPT error audit on gpt_answers_full.csv (57 problems x 10 independent GPT runs = 570 responses)
    ga = pd.read_csv(gp)
    total_runs = len(ga) * 10
    total_correct = int(ga["total_correct"].sum())
    raw_err_rate = 1.0 - total_correct / float(total_runs)
    logic_err_rate = float(ga["logical_errors"].sum()) / float(total_runs)
    arith_err_rate = float(ga["arithmetic_errors"].sum()) / float(total_runs)

    # Derived K5 relative effects and Cohen d
    p2_base = r_p2["vars"]["GPTBase"]["coef"]
    p2_tutor = r_p2["vars"]["GPTTutor"]["coef"]
    p3_base = r_p3["vars"]["GPTBase"]["coef"]
    p3_tutor = r_p3["vars"]["GPTTutor"]["coef"]
    rel_p2_base = p2_base / ctrl_p2_mean
    rel_p2_tutor = p2_tutor / ctrl_p2_mean
    rel_p3_base = p3_base / ctrl_p3_mean
    rel_p3_tutor = p3_tutor / ctrl_p3_mean

    results = {
        "sources": {
            "final_data_csv_sha256": sha_final,
            "gpt_answers_full_csv_sha256": sha_gpt,
        },
        "sample_counts": {
            "full_rows": int(len(df_full)),
            "nonhonors_rows": int(len(df_nonhon)),
            "primary_complete_case_N": int(len(cc)),
            "primary_distinct_students": int(cc["Student ID"].nunique()),
            "primary_classes_G": int(cc["Class"].nunique()),
            "control_part2_mean": ctrl_p2_mean,
            "control_part3_mean": ctrl_p3_mean,
            "control_part3_sd": ctrl_p3_sd,
        },
        "primary_part2_itt": r_p2,
        "primary_part3_itt": r_p3,
        "pretreatment_gpa_balance": r_bal,
        "robustness_part3_compliers": r_p3_comp,
        "robustness_part3_with_honors": r_p3_full,
        "robustness_part3_no_gpa": r_p3_nogpa,
        "robustness_part2_minimal_stratum_fe": r_p2_minfe,
        "robustness_part3_minimal_stratum_fe": r_p3_minfe,
        "robustness_part3_session1_only": r_p3_s1,
        "robustness_part3_student_collapsed": r_p3_coll,
        "relative_and_standardized": {
            "part2_gpt_base_rel": rel_p2_base,
            "part2_gpt_tutor_rel": rel_p2_tutor,
            "part3_gpt_base_rel": rel_p3_base,
            "part3_gpt_base_cohen_d": p3_base / ctrl_p3_sd,
            "part3_gpt_tutor_rel": rel_p3_tutor,
            "part3_gpt_tutor_cohen_d": p3_tutor / ctrl_p3_sd,
        },
        "prior_mapping_to_sd_model": {
            "alpha_ai_canonical": 0.90,
            "alpha_ai_lhs_bounds": [0.65, 1.15],
            "empirical_part2_rel_boost_base": rel_p2_base,
            "implied_alpha_ai_at_x060": rel_p2_base / 0.60,
            "implied_alpha_ai_at_x075": rel_p2_base / 0.75,
            "empirical_part2_rel_boost_ge_040": rel_p2_base >= 0.40,
            "alpha_p_canonical": 0.14,
            "alpha_p_lhs_bounds": [0.10, 0.18],
            "canonical_solo_drop_at_x1": -0.65 * 0.14,
            "empirical_part3_rel_deficit_base": -rel_p3_base,
            "empirical_part3_rel_deficit_ge_010": (-rel_p3_base) >= 0.10,
            "phi_e_canonical": 0.20,
            "phi_e_lhs_bounds": [0.15, 0.25],
            "raw_llm_error_rate_570_runs": raw_err_rate,
            "note": "K5 anchors alpha_ai = r_P2 / x_eff (centered inside LHS [0.65, 1.15]), solo practice drop direction/magnitude alpha_p = 0.14 (LHS [0.10, 0.18]), and raw LLM error prevalence motivating tau_v and p_out (distinct from organizational OOD share phi_e = 0.20, LHS [0.15, 0.25]).",
        },
        "gpt_error_audit": {
            "problems": int(len(ga)),
            "total_responses": int(total_runs),
            "total_correct": int(total_correct),
            "raw_error_rate": raw_err_rate,
            "logical_error_rate": logic_err_rate,
            "arithmetic_error_rate": arith_err_rate,
        },
        "k5_falsifier_checks": {
            "part2_base_lb_ge_010": r_p2["vars"]["GPTBase"]["ci95_wild_boot"][0] >= 0.10,
            "part3_base_abs_deficit_ge_004": (-p3_base) >= 0.04,
            "part3_base_rel_deficit_ge_010": (-rel_p3_base) >= 0.10,
            "part3_base_ci_below_zero": r_p3["vars"]["GPTBase"]["ci95_wild_boot"][1] < 0.0,
            "part3_tutor_abs_lt_004": abs(p3_tutor) < 0.04,
            "balance_gpa_base_p_ge_001": r_bal["vars"]["GPTBase"]["p_wild_boot"] >= 0.01,
            "balance_gpa_tutor_p_ge_001": r_bal["vars"]["GPTTutor"]["p_wild_boot"] >= 0.01,
            "robust_minfe_p3_base_negative": r_p3_minfe["vars"]["GPTBase"]["coef"] < 0.0,
            "robust_s1_p3_base_negative": r_p3_s1["vars"]["GPTBase"]["coef"] < 0.0,
            "robust_collapsed_p3_base_negative": r_p3_coll["vars"]["GPTBase"]["coef"] < 0.0,
        },
    }

    (OUT_DIR / "classroom_prior_reanalysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Export clean CSV summary table for manuscript Table and reproducibility
    rows_csv = []
    for spec_name, obj, outcome in [
        ("Primary_ITT_NonHonors", r_p2, "Part2Tot"),
        ("Primary_ITT_NonHonors", r_p3, "Part3Tot"),
        ("Balance_PreTreatment_GPA", r_bal, "gpa_prev"),
        ("Robustness_Compliers_Only", r_p3_comp, "Part3Tot"),
        ("Robustness_Full_With_Honors", r_p3_full, "Part3Tot"),
        ("Robustness_No_GPA_Control", r_p3_nogpa, "Part3Tot"),
        ("Robustness_Minimal_Stratum_FE", r_p2_minfe, "Part2Tot"),
        ("Robustness_Minimal_Stratum_FE", r_p3_minfe, "Part3Tot"),
        ("Robustness_Session1_Only", r_p3_s1, "Part3Tot"),
        ("Robustness_Student_Collapsed", r_p3_coll, "Part3Tot"),
    ]:
        for arm in ["GPTBase", "GPTTutor"]:
            v = obj["vars"][arm]
            rows_csv.append({
                "specification": spec_name,
                "outcome": outcome,
                "arm": arm,
                "N": obj["N"],
                "G_clusters": obj["G_clusters"],
                "coef": round(v["coef"], 6),
                "se_cluster": round(v["se_cluster"], 6),
                "p_cluster": round(v["p_cluster"], 6),
                "ci95_wild_lo": round(v["ci95_wild_boot"][0], 6),
                "ci95_wild_hi": round(v["ci95_wild_boot"][1], 6),
                "p_wild_boot": round(v["p_wild_boot"], 6),
            })
    pd.DataFrame(rows_csv).to_csv(OUT_DIR / "classroom_prior_regressions.csv", index=False)
    print(json.dumps(results["sample_counts"], indent=2))
    print(json.dumps(results["relative_and_standardized"], indent=2))
    print(json.dumps(results["gpt_error_audit"], indent=2))
    print(json.dumps(results["k5_falsifier_checks"], indent=2))


if __name__ == "__main__":
    main()
