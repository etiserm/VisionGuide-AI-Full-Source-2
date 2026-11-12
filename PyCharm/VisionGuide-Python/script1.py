
import csv
import statistics
from pathlib import Path
from time import perf_counter

import torch
from ultralytics import YOLO


MODEL_PATH = "yolo26n.pt"
IMAGE_PATH = "bus.jpg"
DEVICE = "mps"

WARMUP_RUNS = 3
MEASURED_RUNS = 20
CONFIDENCE = 0.4


def main():
    model = YOLO(MODEL_PATH)

    print("VisionGuide AI - Performance Experiment")
    print("Loading model...")

    print(f"\nWarm-up: {WARMUP_RUNS} runs")

    for _ in range(WARMUP_RUNS):
        model.predict(
            source=IMAGE_PATH,
            device=DEVICE,
            conf=CONFIDENCE,
            verbose=False
        )

    if DEVICE == "mps":
        torch.mps.synchronize()

    print(f"Starting {MEASURED_RUNS} measurements...")

    results_data = []

    for run_number in range(1, MEASURED_RUNS + 1):
        if DEVICE == "mps":
            torch.mps.synchronize()

        start_time = perf_counter()

        results = model.predict(
            source=IMAGE_PATH,
            device=DEVICE,
            conf=CONFIDENCE,
            verbose=False
        )

        if DEVICE == "mps":
            torch.mps.synchronize()

        elapsed_ms = (perf_counter() - start_time) * 1000

        detected_count = sum(len(result.boxes) for result in results)

        results_data.append({
            "run": run_number,
            "total_time_ms": elapsed_ms,
            "detected_objects": detected_count
        })

        print(
            f"Run {run_number:02d}: "
            f"{elapsed_ms:.2f} ms, "
            f"{detected_count} objects"
        )

    # Statistics
    times = [row["total_time_ms"] for row in results_data]

    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    std_time = statistics.stdev(times)

    output_dir = Path("research/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "baseline_experiment.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run",
                "total_time_ms",
                "detected_objects"
            ]
        )
        writer.writeheader()
        writer.writerows(results_data)

    print("\nEXPERIMENT SUMMARY")
    print(f"Mean: {mean_time:.2f} ms")
    print(f"Median: {median_time:.2f} ms")
    print(f"Standard deviation: {std_time:.2f} ms")
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
