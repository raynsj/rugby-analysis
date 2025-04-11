import time
from datetime import datetime
import csv

class PerformanceTracker:
    def __init__(self, csv_file='performance_metrics.csv'):
        self.start_time = time.time()
        self.section_times = {}
        self.csv_file = csv_file

    def start_section(self, section_name):
        """Start timing a section."""
        self.section_times[section_name] = time.time()

    def end_section(self, section_name):
        """End timing a section and calculate elapsed time."""
        if section_name in self.section_times:
            elapsed_time = time.time() - self.section_times[section_name]
            self.section_times[section_name] = round(elapsed_time, 2)
        else:
            print(f"Warning: Section '{section_name}' was not started.")

    def record_metrics(self):
        """Record all metrics to a CSV file."""
        total_time = round(time.time() - self.start_time, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare data for CSV
        row = {'timestamp': timestamp, 'total_time': total_time}
        row.update(self.section_times)

        # Write to CSV
        try:
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=row.keys())
                if file.tell() == 0:  # Write header if file is empty
                    writer.writeheader()
                writer.writerow(row)
            print(f"Metrics recorded to {self.csv_file}")
        except Exception as e:
            print(f"Error writing to CSV: {e}")
