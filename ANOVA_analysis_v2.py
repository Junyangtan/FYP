"""Compare ACO, GA and NSGAII on matched map/cluster problems.

Dependencies: numpy, pandas, scipy, matplotlib (no statsmodels required).
Default: python ANOVA_analysis.py
Custom:  python ANOVA_analysis.py --input analysis_data2.csv --output results

The same (Map, Cluster) must identify the same problem for all algorithms.
Ten runs are averaged per problem/algorithm; they are not 10 independent maps.
All metric values are used unchanged. Lower is assumed better for each metric.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGORITHMS = ["ACO", "GA", "NSGAII"]
METRICS = ["Distance", "Elevation", "Turning"]
ALPHA = 0.05


def holm(pvalues):
    """Holm family-wise correction; retain the input ordering."""
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Cannot adjust undefined p-values.")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def repeated_anova(y):
    """Balanced one-factor RM ANOVA with Greenhouse-Geisser correction.

    Rows are independent problem blocks, columns are algorithms. Covariance
    is centered in contrast space: epsilon = tr(V)^2/((k-1)*tr(V@V)).
    See https://pingouin-stats.org/generated/pingouin.epsilon.html
    """
    n, k = y.shape
    residual = y - y.mean(1, keepdims=True) - y.mean(0) + y.mean()
    ss_error = np.square(residual).sum()
    ss_algorithm = n * np.square(y.mean(0) - y.mean()).sum()
    if ss_error <= np.finfo(float).eps * max(1, ss_algorithm):
        raise ValueError("Degenerate repeated-measures variance; inspect data.")
    df1, df2 = k - 1, (n - 1) * (k - 1)
    f_value = (ss_algorithm / df1) / (ss_error / df2)
    centering = np.eye(k) - np.ones((k, k)) / k
    v = centering @ np.cov(y, rowvar=False) @ centering
    epsilon = float(np.clip(np.trace(v)**2 / ((k-1)*np.trace(v@v)),
                            1/(k-1), 1))
    return dict(N_Blocks=n, F_value=f_value, DF1=df1, DF2=df2,
                GG_Epsilon=epsilon, GG_DF1=epsilon*df1,
                GG_DF2=epsilon*df2,
                p_uncorrected=stats.f.sf(f_value, df1, df2),
                p_GG=stats.f.sf(f_value, epsilon*df1, epsilon*df2))


def load_data(path, expected_runs):
    df = pd.read_csv(path)
    keys = ["Map", "Cluster", "Algorithm", "Run"]
    missing = set(keys + METRICS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[keys+METRICS].isna().any().any() or df.duplicated(keys).any():
        raise ValueError("Missing values or duplicate map/cluster/algorithm/run keys.")
    if set(df.Algorithm) != set(ALGORITHMS):
        raise ValueError(f"Expected algorithms {ALGORITHMS}.")
    if not np.isfinite(df[METRICS].to_numpy(dtype=float)).all():
        raise ValueError("Metrics must be finite numbers.")
    for block, group in df.groupby(["Map", "Cluster"]):
        if set(group.Algorithm) != set(ALGORITHMS):
            raise ValueError(f"Unmatched algorithms in block {block}.")
        run_sets = [set(group.loc[group.Algorithm == a, "Run"]) for a in ALGORITHMS]
        if any(len(s) != expected_runs for s in run_sets) or any(s != run_sets[0] for s in run_sets):
            raise ValueError(f"Inconsistent or incomplete runs in block {block}.")
    for name, group in df.groupby("Map"):
        if group.Cluster.nunique() < 3:
            raise ValueError(f"At least three problem blocks required for map {name}.")
    return df


def analyze(df):
    cm = df.groupby(["Map", "Cluster", "Algorithm"])[METRICS].mean()
    omnibus, pairwise, descriptive, sensitivity, reference, tukey = [], [], [], [], [], []
    scopes = [("Overall", cm)] + [(str(m), cm.loc[[m]]) for m in df.Map.unique()]
    for scope, frame in scopes:
        for metric in METRICS:
            wide = frame[metric].unstack("Algorithm")[ALGORITHMS]
            y = wide.to_numpy()
            n = len(y)
            omnibus.append(dict(Scope=scope, Metric=metric, **repeated_anova(y)))
            friedman = stats.friedmanchisquare(*y.T)
            sensitivity.append(dict(Scope=scope, Metric=metric,
                                    Friedman_Statistic=friedman.statistic,
                                    p_Friedman=friedman.pvalue))
            for i, a in enumerate(ALGORITHMS):
                descriptive.append(dict(Scope=scope, Metric=metric, Algorithm=a,
                                        N_Blocks=n, Mean=y[:, i].mean(),
                                        SD_Across_Cluster_Means=y[:, i].std(ddof=1)))
            for i, j in combinations(range(3), 2):
                a, b = ALGORITHMS[i], ALGORITHMS[j]
                diff = y[:, j] - y[:, i]
                mean, se = float(diff.mean()), float(stats.sem(diff))
                if se == 0:
                    p = 1.0 if mean == 0 else 0.0
                    t = 0.0 if mean == 0 else np.copysign(np.inf, mean)
                else:
                    test = stats.ttest_rel(y[:, j], y[:, i])
                    t, p = test.statistic, test.pvalue
                q = stats.t.ppf(1-ALPHA/2, n-1)
                # Diagnostic/sensitivity results are always exported, never
                # used to switch the primary test to obtain significance.
                wp = 1.0 if np.all(diff == 0) else stats.wilcoxon(diff).pvalue
                pairwise.append(dict(Scope=scope, Metric=metric, Group1=a, Group2=b,
                    N_Blocks=n, Mean_Difference_Group2_minus_Group1=mean,
                    Percent_Difference_Relative_to_Group1=100*mean/y[:, i].mean()
                        if y[:, i].mean() != 0 else np.nan,
                    SE_Difference=se, CI95_Lower_Unadjusted=mean-q*se,
                    CI95_Upper_Unadjusted=mean+q*se, t_value=t, DF=n-1,
                    p_raw=p, p_Wilcoxon=wp,
                    Shapiro_Difference_p=stats.shapiro(diff).pvalue if se else np.nan))
            if scope == "Overall":
                old = stats.f_oneway(*y.T)
                reference.append(dict(Metric=metric, F_value=old.statistic, p_value=old.pvalue))
                # Reproduce conventional independent Tukey for reference only.
                # Export ALL pairs, even when independent ANOVA is nonsignificant.
                th = stats.tukey_hsd(*y.T)
                ci = th.confidence_interval()
                for i, j in combinations(range(3), 2):
                    tukey.append(dict(Metric=metric, Group1=ALGORITHMS[i], Group2=ALGORITHMS[j],
                        Mean_Difference_Group2_minus_Group1=-th.statistic[i, j],
                        p_adj=th.pvalue[i, j], Lower=-ci.high[i, j], Upper=-ci.low[i, j],
                        Significant=bool(th.pvalue[i, j] < ALPHA),
                        Method="Independent Tukey HSD: reference only; ignores matching"))
    omnibus, pairwise, sensitivity = map(pd.DataFrame, (omnibus, pairwise, sensitivity))
    # Conservative families include overall AND map-specific tests:
    # 9 omnibus tests and 27 pairwise tests for the supplied two-map dataset.
    omnibus["p_GG_Holm_All_Scopes"] = holm(omnibus.p_GG)
    omnibus["Significant"] = omnibus.p_GG_Holm_All_Scopes < ALPHA
    pairwise["p_Holm_All_Scopes"] = holm(pairwise.p_raw)
    pairwise["Significant"] = pairwise.p_Holm_All_Scopes < ALPHA
    pairwise["p_Wilcoxon_Holm_All_Scopes"] = holm(pairwise.p_Wilcoxon)
    sensitivity["p_Friedman_Holm_All_Scopes"] = holm(sensitivity.p_Friedman)
    # Bonferroni simultaneous CIs are more conservative than Holm tests.
    q = stats.t.ppf(1-ALPHA/(2*len(pairwise)), pairwise.DF)
    pairwise["CI_Family95_Lower_Bonferroni"] = pairwise.Mean_Difference_Group2_minus_Group1-q*pairwise.SE_Difference
    pairwise["CI_Family95_Upper_Bonferroni"] = pairwise.Mean_Difference_Group2_minus_Group1+q*pairwise.SE_Difference
    return {"cluster_means": cm.reset_index(), "matched_anova": omnibus,
            "paired_comparisons": pairwise, "descriptive_statistics": pd.DataFrame(descriptive),
            "friedman_sensitivity": sensitivity,
            "independent_anova_reference": pd.DataFrame(reference),
            "tukey_independent_reference": pd.DataFrame(tukey)}


def plot_differences(pairwise, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        part = pairwise[(pairwise.Scope == "Overall") & (pairwise.Metric == metric)]
        means = part.Mean_Difference_Group2_minus_Group1.to_numpy()
        lower = part.CI_Family95_Lower_Bonferroni.to_numpy()
        upper = part.CI_Family95_Upper_Bonferroni.to_numpy()
        ax.errorbar(means, np.arange(3), xerr=np.array([means-lower, upper-means]),
                    fmt="o", capsize=5, color="#176b99")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(np.arange(3), [f"{r.Group2} − {r.Group1}" for r in part.itertuples()])
        ax.invert_yaxis()
        ax.set_title(metric)
        ax.set_xlabel("Mean difference (CSV metric units)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(f"Matched cluster differences\n95% simultaneous intervals across all {len(pairwise)} comparisons", fontsize=13)
    fig.savefig(output / "matched_differences.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Output/analysis_data2.csv"))
    parser.add_argument("--output", type=Path, default=Path("Output/Analysis_Matched"))
    parser.add_argument("--expected-runs", type=int, default=10)
    args = parser.parse_args()
    df = load_data(args.input, args.expected_runs)
    results = analyze(df)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(args.output / f"{name}.csv", index=False, float_format="%.12g")
    plot_differences(results["paired_comparisons"], args.output)
    print(f"Validated {len(df)} rows, {len(results['cluster_means'])//3} matched blocks.")
    print(results["matched_anova"].to_string(index=False))
    print("Pairwise comparisons are always exported; primary method is paired t + Holm.")
    print("Independent Tukey is a reference output, not the matched post-hoc test.")
    print("Inference assumes independent problem blocks and identical problems across algorithms.")
    print("Results describe these benchmark maps; statistical significance does not select priorities.")
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
