#!/usr/bin/env python3
"""
results.py

Script to analyze and plot robot counting results.
1. CSV robot counting results for 3 environments, 5 attempts each.
2. Loads the CSV into a pandas DataFrame.
3. Generates summary statistics and multiple plots (bar chart, line plot).
4. Displays the plots interactively and also saves them as PNG files.

By Kev
"""

import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # -------------------------------------------------------------------------
    # 1) Embedded CSV data: 15 rows (3 env x 5 attempts).
    #    The columns are: environment,attempt,red,blue,green,total,time
    #    'time' is total time in seconds. 'total' is sum of red,blue,green (possibly double counts).
    # -------------------------------------------------------------------------
    csv_data = """environment,attempt,red,blue,green,total,time
Env1,1,74,82,92,248,154.01
Env1,2,125,100,66,291,169.17
Env1,3,100,57,68,225,185.91
Env1,4,93,69,72,234,162.31
Env1,5,120,75,49,244,165.51
Env2,1,105,58,28,191,225.56
Env2,2,46,51,35,132,170.36
Env2,3,35,44,40,119,168.61
Env2,4,55,44,57,156,181.75
Env2,5,59,24,54,137,187.44
Env3,1,47,15,14,76,162.21
Env3,2,14,47,22,83,162.61
Env3,3,43,32,32,94,160.01
Env3,4,28,34,35,97,173.96
Env3,5,36,36,35,107,165.06
"""

    # -------------------------------------------------------------------------
    # 2) Parse the CSV into a pandas DataFrame (no external file needed).
    # -------------------------------------------------------------------------
    df = pd.read_csv(io.StringIO(csv_data))
    # Convert numeric columns just in case
    numeric_cols = ['red','blue','green','total','time']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # -------------------------------------------------------------------------
    # 3) Print summary stats by environment
    # -------------------------------------------------------------------------
    group_env = df.groupby('environment')
    summary = group_env[['red','blue','green','total','time']].mean()
    print("\n=== Average Values by Environment ===\n")
    print(summary.round(2))
    print()

    # -------------------------------------------------------------------------
    # 4) Make a bar chart of average color counts (red,blue,green) by environment
    # -------------------------------------------------------------------------
    avg_colors = group_env[['red','blue','green']].mean().reset_index()
    # Melt for easier plotting in seaborn
    melted_colors = avg_colors.melt(id_vars='environment', value_vars=['red','blue','green'],
                                    var_name='color', value_name='count')
    
    plt.figure(figsize=(6,4))
    sns.barplot(data=melted_colors, x='environment', y='count', hue='color', palette='Set2')
    plt.title("Average Color Counts by Environment")
    plt.ylabel("Mean Count")
    plt.xlabel("Environment")
    plt.legend(title="Color")
    plt.tight_layout()
    plt.savefig("avg_color_counts_by_env.png")
    plt.show()

    # -------------------------------------------------------------------------
    # 5) Make a line plot of total time across attempts for each environment
    # -------------------------------------------------------------------------
    plt.figure(figsize=(6,4))
    sns.lineplot(data=df, x='attempt', y='time', hue='environment', marker='o')
    plt.title("Total Time per Attempt for Each Environment")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Attempt")
    plt.tight_layout()
    plt.savefig("time_per_attempt_lineplot.png")
    plt.show()

    # Done
    print("Plots saved: avg_color_counts_by_env.png, time_per_attempt_lineplot.png")
    print("Script completed successfully.")

if __name__ == "__main__":
    main()
