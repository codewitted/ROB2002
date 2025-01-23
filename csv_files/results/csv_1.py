import csv
import matplotlib.pyplot as plt
import numpy as np

def read_csv(csv_path):
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def parse_data(data):
    """Group rows by method (Baseline/Enhanced), compute average metrics."""
    baseline_runs = [row for row in data if row['method'] == 'Baseline']
    enhanced_runs = [row for row in data if row['method'] == 'Enhanced']

    def avg(vals):
        return sum(vals)/len(vals) if vals else 0

    # Convert string fields to float
    def to_float(x):
        try:
            return float(x)
        except:
            return 0.0

    # Example metrics
    baseline_detected = avg([to_float(r['total_detected']) for r in baseline_runs])
    baseline_misses   = avg([to_float(r['misses']) for r in baseline_runs])
    baseline_time     = avg([to_float(r['runtime_s']) for r in baseline_runs])

    enhanced_detected = avg([to_float(r['total_detected']) for r in enhanced_runs])
    enhanced_misses   = avg([to_float(r['misses']) for r in enhanced_runs])
    enhanced_time     = avg([to_float(r['runtime_s']) for r in enhanced_runs])

    return (baseline_detected, baseline_misses, baseline_time,
            enhanced_detected, enhanced_misses, enhanced_time)

def plot_results(env_name, baseline, enhanced):
    # baseline & enhanced are tuples: (detected, misses, time)
    x_labels = ['Detected', 'Misses', 'Time (s)']
    baseline_vals = baseline
    enhanced_vals = enhanced

    x = np.arange(len(x_labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, baseline_vals, width, label='Baseline')
    ax.bar(x + width/2, enhanced_vals, width, label='Enhanced')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title(f'Results for {env_name}')
    ax.legend()

    plt.show()

if __name__ == '__main__':
    # For example, read environment 1's CSV
    env1_data = read_csv('env1_results.csv')
    env2_data = read_csv('env2_results.csv')
    env3_data = read_csv('env3_results.csv')

    (bdet1, bmis1, btime1, edet1, emis1, etime1) = parse_data(env1_data)
    (bdet2, bmis2, btime2, edet2, emis2, etime2) = parse_data(env2_data)
    (bdet3, bmis3, btime3, edet3, emis3, etime3) = parse_data(env3_data)

    plot_results("Environment 1", (bdet1, bmis1, btime1), (edet1, emis1, etime1))
    plot_results("Environment 2", (bdet2, bmis2, btime2), (edet2, emis2, etime2))
    plot_results("Environment 3", (bdet3, bmis3, btime3), (edet3, emis3, etime3))
