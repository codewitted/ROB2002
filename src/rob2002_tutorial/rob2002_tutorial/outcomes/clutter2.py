#!/usr/bin/env python3
"""
clutter_graph.py

Creates a single figure with 4 grouped bar charts showing:
- Average Total Detected
- Average Duplicates
- Average Misses
- Average RunTime (seconds)

for Baseline vs Enhanced across three environments (Env1, Env2, Env3).
Data obtained from final tables.

By Kev
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Average data from your summary tables:
    data = [
        # Environment,   Method,    TotalDetected, Duplicates, Misses,   RunTime
        ["Env1 (Easy)",    "Baseline",  5.7,         0.7,       0.7,      80.7],
        ["Env1 (Easy)",    "Enhanced",  6.0,         0.0,       0.0,      96.7],
        ["Env2 (Medium)",  "Baseline",  7.3,         0.7,       1.0,     120.8],
        ["Env2 (Medium)",  "Enhanced",  8.0,         0.0,       0.0,     135.5],
        ["Env3 (Hard)",    "Baseline",  8.3,         1.3,       1.7,     156.1],
        ["Env3 (Hard)",    "Enhanced", 10.0,         0.3,       0.0,     174.5],
    ]

    df = pd.DataFrame(data, columns=[
        "Environment", "Method", "TotalDetected", "Duplicates", "Misses", "RunTime"
    ])

    # Set a nice style
    sns.set(style="whitegrid", context="talk")
    
    # Create subplots: 2 rows x 2 cols
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    # 1) Total Detected
    sns.barplot(
        data=df, x="Environment", y="TotalDetected", hue="Method",
        palette="Set2", ax=axs[0,0]
    )
    axs[0,0].set_title("Average Total Detected")
    axs[0,0].set_ylabel("Count")
    axs[0,0].legend(loc="best")

    # 2) Duplicates
    sns.barplot(
        data=df, x="Environment", y="Duplicates", hue="Method",
        palette="Set2", ax=axs[0,1]
    )
    axs[0,1].set_title("Average Duplicates")
    axs[0,1].set_ylabel("Duplicates")
    axs[0,1].legend(loc="best")

    # 3) Misses
    sns.barplot(
        data=df, x="Environment", y="Misses", hue="Method",
        palette="Set2", ax=axs[1,0]
    )
    axs[1,0].set_title("Average Misses")
    axs[1,0].set_ylabel("Misses")
    axs[1,0].legend(loc="best")

    # 4) RunTime
    sns.barplot(
        data=df, x="Environment", y="RunTime", hue="Method",
        palette="Set2", ax=axs[1,1]
    )
    axs[1,1].set_title("Average RunTime (s)")
    axs[1,1].set_ylabel("Seconds")
    axs[1,1].legend(loc="best")

    plt.tight_layout()
    plt.savefig("clutter_comparison.png")

    plt.show()

    print("Plot saved as clutter_comparison.png")

if __name__ == "__main__":
    main()
