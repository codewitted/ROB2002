import csv
import matplotlib.pyplot as plt
import statistics

def analyze_experiment(log_file, true_count):
    """
    log_file: Path to the CSV file produced by the counting node
    true_count: The known number of objects (ground truth)
    """
    timestamps = []
    counts = []
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))
            counts.append(int(row['count']))

    final_count = counts[-1] if counts else 0
    accuracy = final_count / true_count if true_count > 0 else 0

    print(f'File: {log_file}')
    print(f'Final Count: {final_count}')
    print(f'Ground Truth: {true_count}')
    print(f'Accuracy: {accuracy*100:.2f}%')

    # Basic plot of count vs. time
    plt.figure()
    plt.plot(timestamps, counts, label='Detected Count')
    plt.axhline(y=true_count, color='r', linestyle='--', label='Ground Truth')
    plt.xlabel('Time (s)')
    plt.ylabel('Object Count')
    plt.title(f'Counting Performance: {log_file}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Example usage
    log_files = [
        'object_count_log_scenario1.csv',
        'object_count_log_scenario2.csv'
    ]
    # Suppose scenario1 had 5 objects, scenario2 had 8, etc.
    true_counts = [5, 8]

    for lf, tc in zip(log_files, true_counts):
        analyze_experiment(lf, tc)
