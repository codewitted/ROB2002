#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def main():
    environments = ["Env1", "Env2", "Env3"]
    x = np.arange(len(environments))  # [0, 1, 2]

    red_vals   = [102.4, 48.75, 33.6]
    blue_vals  = [76.6,  44.2,  30.2]
    green_vals = [69.4,  42.8,  27.6]
    total_vals = [248.4, 147,   91.4]
    time_vals  = [167.382, 186.744, 164.77]

    plt.figure(figsize=(7,5))
    plt.title("Environment Averages")

    # Plot each metric as a separate line
    plt.plot(x, red_vals,   label="Red",   marker='o', color='red')
    plt.plot(x, blue_vals,  label="Blue",  marker='o', color='blue')
    plt.plot(x, green_vals, label="Green", marker='o', color='green')
    plt.plot(x, total_vals, label="Total", marker='o', color='darkred')
    plt.plot(x, time_vals,  label="Time taken (s)", marker='o', color='darkblue')

    # Label the x-axis with environment names
    plt.xticks(x, environments)

    # Set to suitable y-limit
    plt.ylim(0, 300)

    # Add axis labels
    plt.xlabel("Environment")
    plt.ylabel("Value / Time (seconds)")

    # Add a legend
    plt.legend()

    # Creates a png
    plt.tight_layout()
    plt.savefig("environment_averages.png")
    plt.show()

if __name__ == "__main__":
    main()
