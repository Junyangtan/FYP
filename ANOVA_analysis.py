import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# =====================================================
# SETTINGS
# =====================================================

INPUT_FILE = "Output/analysis_data1.csv"

OUTPUT_FOLDER = "Output/Analysis"


ALGORITHMS = [
    "ACO",
    "GA",
    "NSGAII"
]


MAPS = [
    "Map1",
    "Map2"
]


METRICS = [
    "Distance",
    "Elevation",
    "Turning"
]


# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(INPUT_FILE)


print("\n======================================")
print("DATA LOADED")
print("======================================")

print(df.head())


# =====================================================
# CHECK REQUIRED COLUMNS
# =====================================================

required_columns = [
    "Map",
    "Cluster",
    "Algorithm",
    "Run",
    "Distance",
    "Elevation",
    "Turning"
]


missing_columns = [
    col for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing columns: "
        +
        str(missing_columns)
    )


# =====================================================
# CHECK DATA
# =====================================================

print("\nAlgorithms found:")
print(df["Algorithm"].unique())


print("\nMaps found:")
print(df["Map"].unique())


print("\nNumber of raw rows:")
print(len(df))


# =====================================================
# CHECK RUN COUNT
# =====================================================

run_count = (
    df
    .groupby(
        [
            "Map",
            "Cluster",
            "Algorithm"
        ]
    )["Run"]
    .nunique()
    .reset_index(
        name="Number_of_Runs"
    )
)


print("\n======================================")
print("RUN COUNT CHECK")
print("======================================")

print(run_count)


incorrect_runs = run_count[
    run_count["Number_of_Runs"] != 10
]


if not incorrect_runs.empty:

    print("\nWARNING:")
    print(
        "Some Map/Cluster/Algorithm combinations "
        "do not contain exactly 10 runs."
    )

    print(incorrect_runs)


# =====================================================
# STEP 1
# CALCULATE MEAN OF 10 RUNS FOR EACH CLUSTER
# =====================================================

cluster_mean = (
    df
    .groupby(
        [
            "Map",
            "Cluster",
            "Algorithm"
        ]
    )[METRICS]
    .mean()
    .reset_index()
)


cluster_mean.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "cluster_mean_results.csv"
    ),
    index=False
)


print("\n======================================")
print("CLUSTER MEANS")
print("======================================")

print(cluster_mean)


# =====================================================
# SHOW NUMBER OF CLUSTERS
# =====================================================

print("\n======================================")
print("NUMBER OF CLUSTER MEANS")
print("======================================")


for algo in ALGORITHMS:

    algo_data = cluster_mean[
        cluster_mean["Algorithm"] == algo
    ]

    print(
        algo,
        "=",
        len(algo_data),
        "combined cluster values"
    )


# =====================================================
# STEP 2
# MERGE MAP1 + MAP2
# =====================================================

combined_data = cluster_mean.copy()


combined_data.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "combined_cluster_results.csv"
    ),
    index=False
)


# =====================================================
# STEP 3
# OVERALL MEAN + STANDARD DEVIATION
# =====================================================

summary_rows = []


for algo in ALGORITHMS:

    algo_data = combined_data[
        combined_data["Algorithm"] == algo
    ]


    for metric in METRICS:

        values = algo_data[metric]


        summary_rows.append(
            [
                algo,
                metric,
                len(values),
                values.mean(),
                values.std(ddof=1)
            ]
        )


overall_summary = pd.DataFrame(
    summary_rows,
    columns=[
        "Algorithm",
        "Metric",
        "N",
        "Mean",
        "Standard_Deviation"
    ]
)


overall_summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "overall_summary.csv"
    ),
    index=False
)


print("\n======================================")
print("OVERALL SUMMARY")
print("======================================")

print(overall_summary)


# =====================================================
# P-VALUE FORMAT
# =====================================================

def format_p_value(p):

    if p < 0.00001:

        return "<0.00001"

    return f"{p:.5f}"


# =====================================================
# SIGNIFICANCE STAR
# =====================================================

def significance_symbol(p):

    if p < 0.001:

        return "***"

    elif p < 0.01:

        return "**"

    elif p < 0.05:

        return "*"

    else:

        return "ns"


# =====================================================
# STEP 4
# ONE-WAY ANOVA
# =====================================================

anova_results = []

tukey_results = []

tukey_plot_results = {}


for metric in METRICS:


    print("\n\n======================================")
    print("ANOVA -", metric)
    print("======================================")


    groups = []


    for algo in ALGORITHMS:


        values = combined_data[
            combined_data["Algorithm"] == algo
        ][metric].dropna()


        groups.append(values)


        print(
            algo,
            "N =",
            len(values),
            "Mean =",
            round(values.mean(), 5),
            "STD =",
            round(values.std(ddof=1), 5)
        )


    # -------------------------------------------------
    # ANOVA
    # -------------------------------------------------

    F_value, p_value = f_oneway(
        *groups
    )


    print("\nANOVA result:")


    print(
        "F-value =",
        round(F_value, 5)
    )


    print(
        "p-value =",
        format_p_value(p_value)
    )


    if p_value < 0.05:

        result = "Significant"

    else:

        result = "Not Significant"


    print(
        "Result =",
        result
    )


    anova_results.append(
        [
            metric,
            F_value,
            p_value,
            result
        ]
    )


    # =================================================
    # TUKEY HSD
    # =================================================

    if p_value < 0.05:


        tukey = pairwise_tukeyhsd(
            endog=combined_data[metric],
            groups=combined_data["Algorithm"],
            alpha=0.05
        )


        group_indexes_1 = (
            tukey
            ._multicomp
            .pairindices[0]
        )


        group_indexes_2 = (
            tukey
            ._multicomp
            .pairindices[1]
        )


        group_names = tukey.groupsunique


        plot_pairs = []


        print("\nTukey HSD:")
        print(tukey)


        for (
            idx1,
            idx2,
            mean_diff,
            pair_p,
            confidence_interval,
            reject
        ) in zip(
            group_indexes_1,
            group_indexes_2,
            tukey.meandiffs,
            tukey.pvalues,
            tukey.confint,
            tukey.reject
        ):


            group1 = group_names[idx1]

            group2 = group_names[idx2]


            lower = confidence_interval[0]

            upper = confidence_interval[1]


            # =================================================
            # SAVE EACH TUKEY COMPARISON AS ONE CSV ROW
            # =================================================

            tukey_results.append(
                [
                    metric,
                    group1,
                    group2,
                    mean_diff,
                    format_p_value(pair_p),
                    lower,
                    upper,
                    bool(reject)
                ]
            )


            # Keep raw p-value for graph
            plot_pairs.append(
                {
                    "group1": group1,
                    "group2": group2,
                    "p_value": pair_p,
                    "reject": bool(reject)
                }
            )


        tukey_plot_results[metric] = plot_pairs


# =====================================================
# SAVE ANOVA RESULTS
# =====================================================

anova_df = pd.DataFrame(
    anova_results,
    columns=[
        "Metric",
        "F_value",
        "p_value",
        "Result"
    ]
)


anova_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "anova_results.csv"
    ),
    index=False
)


# =====================================================
# SAVE TUKEY RESULTS
# AS NORMAL ROWS AND COLUMNS
# =====================================================

tukey_df = pd.DataFrame(
    tukey_results,
    columns=[
        "Metric",
        "Group1",
        "Group2",
        "Mean_Difference",
        "p_adj",
        "Lower",
        "Upper",
        "Significant"
    ]
)


tukey_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "tukey_results.csv"
    ),
    index=False
)


print("\n======================================")
print("TUKEY RESULTS")
print("======================================")

print(tukey_df)


# =====================================================
# FUNCTION:
# ADD SIGNIFICANCE BRACKET
# =====================================================

def add_significance_bracket(
    ax,
    x1,
    x2,
    y,
    height,
    text
):


    ax.plot(
        [
            x1,
            x1,
            x2,
            x2
        ],
        [
            y,
            y + height,
            y + height,
            y
        ],
        linewidth=1.2
    )


    ax.text(
        (x1 + x2) / 2,
        y + height,
        text,
        ha="center",
        va="bottom",
        fontsize=9
    )


# =====================================================
# STEP 5
# MAIN OVERALL GRAPH
# =====================================================

def generate_overall_graph(metric):


    fig, ax = plt.subplots(
        figsize=(9, 7)
    )


    means = []

    stds = []


    for algo in ALGORITHMS:


        row = overall_summary[
            (overall_summary["Algorithm"] == algo)
            &
            (overall_summary["Metric"] == metric)
        ]


        means.append(
            row["Mean"].values[0]
        )


        stds.append(
            row["Standard_Deviation"].values[0]
        )


    x = np.arange(
        len(ALGORITHMS)
    )


    ax.bar(
        x,
        means,
        width=0.55,
        yerr=stds,
        capsize=6
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        ALGORITHMS
    )


    ax.set_xlabel(
        "Algorithm"
    )


    ax.set_ylabel(
        metric
    )


    ax.set_title(
        f"Overall {metric} Comparison"
    )


    # =================================================
    # Overall ANOVA p-value
    # =================================================

    p_anova = anova_df[
        anova_df["Metric"] == metric
    ]["p_value"].values[0]


    ax.text(
        0.02,
        0.97,
        (
            "ANOVA: "
            +
            significance_symbol(p_anova)
            +
            "  p="
            +
            format_p_value(p_anova)
        ),
        transform=ax.transAxes,
        verticalalignment="top"
    )


    # =================================================
    # Tukey pairwise brackets
    # =================================================

    if metric in tukey_plot_results:


        max_height = max(
            np.array(means)
            +
            np.nan_to_num(
                np.array(stds)
            )
        )


        bracket_gap = (
            max_height * 0.08
        )


        bracket_height = (
            max_height * 0.025
        )


        pair_level = 0


        for pair in tukey_plot_results[metric]:


            group1 = pair["group1"]

            group2 = pair["group2"]

            pair_p = pair["p_value"]


            x1 = ALGORITHMS.index(
                group1
            )


            x2 = ALGORITHMS.index(
                group2
            )


            y = (
                max_height
                +
                bracket_gap
                +
                pair_level * bracket_gap
            )


            label = (
                significance_symbol(pair_p)
                +
                "  p="
                +
                format_p_value(pair_p)
            )


            add_significance_bracket(
                ax,
                x1,
                x2,
                y,
                bracket_height,
                label
            )


            pair_level += 1


        ax.set_ylim(
            0,
            max_height
            +
            bracket_gap * 4.5
        )


    plt.tight_layout()


    filename = (
        "overall_"
        +
        metric.lower()
        +
        "_comparison.png"
    )


    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        filename,
        "saved"
    )


# =====================================================
# GENERATE MAIN OVERALL GRAPHS
# =====================================================

for metric in METRICS:

    generate_overall_graph(
        metric
    )


# =====================================================
# STEP 6
# MAP1 vs MAP2 GRAPH
# Visualization only
# =====================================================

def generate_map_comparison_graph(metric):


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    x = np.arange(
        len(ALGORITHMS)
    )


    width = 0.35


    map1_means = []

    map2_means = []

    map1_stds = []

    map2_stds = []


    for algo in ALGORITHMS:


        map1_values = cluster_mean[
            (cluster_mean["Map"] == "Map1")
            &
            (cluster_mean["Algorithm"] == algo)
        ][metric]


        map2_values = cluster_mean[
            (cluster_mean["Map"] == "Map2")
            &
            (cluster_mean["Algorithm"] == algo)
        ][metric]


        map1_means.append(
            map1_values.mean()
        )


        map2_means.append(
            map2_values.mean()
        )


        map1_stds.append(
            map1_values.std(ddof=1)
        )


        map2_stds.append(
            map2_values.std(ddof=1)
        )


    ax.bar(
        x - width / 2,
        map1_means,
        width,
        yerr=map1_stds,
        capsize=5,
        label="Map1"
    )


    ax.bar(
        x + width / 2,
        map2_means,
        width,
        yerr=map2_stds,
        capsize=5,
        label="Map2"
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        ALGORITHMS
    )


    ax.set_xlabel(
        "Algorithm"
    )


    ax.set_ylabel(
        metric
    )


    ax.set_title(
        f"{metric} Comparison by Map"
    )


    ax.legend()


    plt.tight_layout()


    filename = (
        "map_"
        +
        metric.lower()
        +
        "_comparison.png"
    )


    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        filename,
        "saved"
    )


# =====================================================
# GENERATE MAP-SPLIT GRAPHS
# =====================================================

for metric in METRICS:

    generate_map_comparison_graph(
        metric
    )


# =====================================================
# FINISH
# =====================================================

print("\n======================================")
print("ANALYSIS COMPLETE")
print("======================================")


print(
    "All results saved in:",
    OUTPUT_FOLDER
)