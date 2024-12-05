import csv


def write_results_to_csv(output_path, results):
    """
    Write classification results to a CSV file.

    Args:
        output_path (str): Path to the output CSV file.
        results (list): List of tuples (image_path, label).
    """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Label"])  # Header
        writer.writerows(results)
