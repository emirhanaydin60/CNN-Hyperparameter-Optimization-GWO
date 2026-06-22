import argparse
import csv
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def read_text(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return file_handle.read()


def write_text(path, content):
    with open(path, "w", encoding="utf-8") as file_handle:
        file_handle.write(content)


def write_csv(path, fieldnames, rows):
    with open(path, mode="w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_number(value):
    value = value.strip()
    if value.lower() in {"none", "null", ""}:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def parse_key_value_text(path):
    if not os.path.exists(path):
        return {}
    data = {}
    for line in read_text(path).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        data[key] = parse_number(value)
    return data


def parse_final_results(path):
    data = parse_key_value_text(path)
    return {
        "population_size": int(data.get("population_size", 0) or 0),
        "iteration_count": int(data.get("iteration_count", 0) or 0),
        "baseline_test_accuracy": float(data.get("baseline_test_accuracy", 0.0) or 0.0),
        "gwo_test_accuracy": float(data.get("gwo_test_accuracy", 0.0) or 0.0),
        "accuracy_improvement_points": float(data.get("accuracy_improvement_(percentage_points)", 0.0) or 0.0),
        "relative_improvement": float(data.get("relative_improvement_(%)", 0.0) or 0.0),
        "total_evaluations": int(data.get("total_evaluations", 0) or 0),
        "unique_solutions": int(data.get("unique_solutions", 0) or 0),
        "cache_hits": int(data.get("cache_hits", 0) or 0),
        "cache_misses": int(data.get("cache_misses", 0) or 0),
        "cache_reuse_rate": float(data.get("cache_reuse_rate_(%)", 0.0) or 0.0),
        "total_optimization_time": float(data.get("total_optimization_time_(seconds)", 0.0) or 0.0),
        "total_training_time": float(data.get("total_training_time_(seconds)", 0.0) or 0.0),
    }


def parse_best_config(path):
    data = parse_key_value_text(path)
    return {
        "shared_conv_kernel_size": data.get("shared_conv_kernel_size"),
        "base_filters": data.get("base_filters"),
        "dilation": data.get("dilation"),
        "final_neurons": data.get("final_neurons"),
        "dropout": data.get("dropout"),
        "learning_rate": data.get("learning_rate"),
        "batch_size": data.get("batch_size"),
        "se_ratio": data.get("se_ratio"),
        "validation_accuracy": data.get("validation_accuracy"),
    }


def load_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def find_run_dirs(results_dir):
    run_dirs = []
    for entry in sorted(os.listdir(results_dir)):
        candidate = os.path.join(results_dir, entry)
        if os.path.isdir(candidate) and entry.startswith("run_"):
            run_dirs.append(candidate)
    if run_dirs:
        return run_dirs
    return [results_dir]


def read_search_history(path):
    if not os.path.exists(path):
        return []
    with open(path, mode="r", newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
    return rows


def build_iteration_summaries(search_rows):
    grouped = defaultdict(list)
    for row in search_rows:
        try:
            iteration = int(row.get("iteration", 0))
        except ValueError:
            continue
        grouped[iteration].append(row)

    iteration_summaries = []
    for iteration in sorted(grouped):
        rows = grouped[iteration]
        fitness_values = [float(row.get("fitness", 0.0) or 0.0) for row in rows]
        keys = [
            row.get("combination_key")
            or "|".join(
                [
                    str(row.get("shared_conv_kernel_size", "")),
                    str(row.get("base_filters", "")),
                    str(row.get("dilation", "")),
                    str(row.get("final_neurons", "")),
                    str(row.get("dropout", "")),
                    str(row.get("learning_rate", "")),
                    str(row.get("batch_size", "")),
                    str(row.get("se_ratio", "")),
                ]
            )
            for row in rows
        ]
        unique_solution_count = len(set(keys))
        total_count = len(rows)
        repeat_rate = 1.0 - (unique_solution_count / total_count) if total_count else 0.0
        average_fitness = sum(fitness_values) / total_count if total_count else 0.0
        best_fitness = max(fitness_values) if fitness_values else 0.0
        iteration_summaries.append(
            {
                "iteration": iteration,
                "unique_solution_count": unique_solution_count,
                "repeat_rate": repeat_rate,
                "average_fitness": average_fitness,
                "best_fitness": best_fitness,
            }
        )
    return iteration_summaries


def aggregate_stats(runs):
    baseline = [run["baseline_test_accuracy"] for run in runs]
    gwo = [run["gwo_test_accuracy"] for run in runs]
    improvements = [run["accuracy_improvement_points"] for run in runs]
    mean_baseline = sum(baseline) / len(baseline) if baseline else 0.0
    mean_gwo = sum(gwo) / len(gwo) if gwo else 0.0
    mean_improvement = sum(improvements) / len(improvements) if improvements else 0.0
    baseline_std = (sum((value - mean_baseline) ** 2 for value in baseline) / len(baseline)) ** 0.5 if baseline else 0.0
    gwo_std = (sum((value - mean_gwo) ** 2 for value in gwo) / len(gwo)) ** 0.5 if gwo else 0.0
    improvement_std = (sum((value - mean_improvement) ** 2 for value in improvements) / len(improvements)) ** 0.5 if improvements else 0.0
    return {
        "Average Accuracy": {"mean": mean_gwo, "std": gwo_std},
        "Baseline Accuracy": {"mean": mean_baseline, "std": baseline_std},
        "Average Improvement": {"mean": mean_improvement, "std": improvement_std},
    }


def timing_stats(runs, total_experiment_time):
    optimization_times = [run["total_optimization_time"] for run in runs]
    training_times = [run["total_training_time"] for run in runs]
    baseline_times = [run.get("baseline_training_time", 0.0) for run in runs]
    mean_opt = sum(optimization_times) / len(optimization_times) if optimization_times else 0.0
    mean_train = sum(training_times) / len(training_times) if training_times else 0.0
    mean_base = sum(baseline_times) / len(baseline_times) if baseline_times else 0.0
    opt_std = (sum((value - mean_opt) ** 2 for value in optimization_times) / len(optimization_times)) ** 0.5 if optimization_times else 0.0
    train_std = (sum((value - mean_train) ** 2 for value in training_times) / len(training_times)) ** 0.5 if training_times else 0.0
    base_std = (sum((value - mean_base) ** 2 for value in baseline_times) / len(baseline_times)) ** 0.5 if baseline_times else 0.0
    return {
        "GWO Optimization Time": {"mean": mean_opt, "std": opt_std},
        "Baseline Training Time": {"mean": mean_base, "std": base_std},
        "Final GWO Training Time": {"mean": mean_train, "std": train_std},
        "Total Experiment Time": total_experiment_time,
    }


def cache_stats(runs):
    total_evaluations = sum(run["total_evaluations"] for run in runs)
    cache_hits = sum(run["cache_hits"] for run in runs)
    cache_misses = sum(run["cache_misses"] for run in runs)
    cache_reuse_rate = (cache_hits / total_evaluations * 100.0) if total_evaluations else 0.0
    return {
        "total_evaluations": total_evaluations,
        "real_training_count": cache_misses,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "saved_training_count": cache_hits,
        "saved_estimated_time": 0.0,
        "cache_reuse_rate": cache_reuse_rate,
    }


def write_statistics_file(path, stats):
    lines = []
    for key, value in stats.items():
        if isinstance(value, dict):
            lines.append(f"{key}: mean={value['mean']:.6f}, std={value['std']:.6f}")
        else:
            lines.append(f"{key}: {value:.6f}")
    write_text(path, "\n".join(lines) + "\n")


def write_timing_file(path, stats):
    lines = []
    for key, value in stats.items():
        if isinstance(value, dict):
            lines.append(f"{key}: mean={value['mean']:.2f}, std={value['std']:.2f}")
        else:
            lines.append(f"{key}: {value:.2f}")
    write_text(path, "\n".join(lines) + "\n")


def write_cache_statistics_file(path, stats):
    content = [
        f"Total Evaluations: {stats['total_evaluations']}",
        f"Real Training Count: {stats['real_training_count']}",
        f"Cache Hits: {stats['cache_hits']}",
        f"Cache Misses: {stats['cache_misses']}",
        f"Saved Training Count: {stats['saved_training_count']}",
        f"Saved Estimated Time (seconds): {stats['saved_estimated_time']:.6f}",
        f"Cache Reuse Rate (%): {stats['cache_reuse_rate']:.2f}",
    ]
    write_text(path, "\n".join(content) + "\n")


def write_convergence_file(path, iteration_summaries, unique_solutions_evaluated):
    lines = ["iteration,best_fitness,average_fitness,unique_solution_count"]
    global_bests = []
    for summary in iteration_summaries:
        global_bests.append(summary["best_fitness"] if not global_bests else max(global_bests[-1], summary["best_fitness"]))
        lines.append(f"{summary['iteration']},{summary['best_fitness']:.6f},{summary['average_fitness']:.6f},{summary['unique_solution_count']}")
    if iteration_summaries:
        first_fitness = global_bests[0]
        last_fitness = global_bests[-1]
        fitness_increase = last_fitness - first_fitness
        convergence_percent = (fitness_increase / abs(first_fitness) * 100.0) if first_fitness != 0 else 0.0
    else:
        first_fitness = last_fitness = fitness_increase = convergence_percent = 0.0
    lines.extend(
        [
            "",
            f"Unique Solutions Evaluated: {unique_solutions_evaluated}",
            f"first_iter_fitness: {first_fitness:.6f}",
            f"last_iter_fitness: {last_fitness:.6f}",
            f"fitness_increase: {fitness_increase:.6f}",
            f"convergence_percent: {convergence_percent:.2f}",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def write_diversity_file(path, iteration_summaries):
    lines = ["iteration,unique_solution_count,repeat_rate,average_fitness,best_fitness"]
    for summary in iteration_summaries:
        lines.append(f"{summary['iteration']},{summary['unique_solution_count']},{summary['repeat_rate']:.6f},{summary['average_fitness']:.6f},{summary['best_fitness']:.6f}")
    write_text(path, "\n".join(lines) + "\n")


def analyze_confusion_matrix(confusion_matrix, class_names):
    from metrics import normalize_confusion_matrix

    normalized = normalize_confusion_matrix(confusion_matrix)
    class_accuracies = np.diag(normalized)

    confused_pairs = []
    for true_index in range(confusion_matrix.shape[0]):
        for pred_index in range(confusion_matrix.shape[1]):
            if true_index == pred_index:
                continue
            count = int(confusion_matrix[true_index, pred_index])
            if count > 0:
                confused_pairs.append((count, class_names[true_index], class_names[pred_index]))
    confused_pairs.sort(key=lambda item: item[0], reverse=True)

    best_class_index = int(np.argmax(class_accuracies))
    worst_class_index = int(np.argmin(class_accuracies))

    return {
        "top_confusions": confused_pairs[:5],
        "best_class": (class_names[best_class_index], float(class_accuracies[best_class_index])),
        "worst_class": (class_names[worst_class_index], float(class_accuracies[worst_class_index])),
    }


def write_confusion_analysis_file(path, confusion_analysis):
    lines = ["Top 5 Confused Class Pairs:"]
    for count, true_name, pred_name in confusion_analysis["top_confusions"]:
        lines.append(f"{true_name} -> {pred_name}: {count}")
    lines.append(f"Best Accuracy Class: {confusion_analysis['best_class'][0]} ({confusion_analysis['best_class'][1]:.6f})")
    lines.append(f"Worst Accuracy Class: {confusion_analysis['worst_class'][0]} ({confusion_analysis['worst_class'][1]:.6f})")
    write_text(path, "\n".join(lines) + "\n")


def rebuild_training_curves(results_dir):
    accuracy_path = os.path.join(results_dir, "accuracy_curve.png")
    loss_path = os.path.join(results_dir, "loss_curve.png")
    output_path = os.path.join(results_dir, "training_curves.png")

    if os.path.exists(output_path):
        return output_path

    if not (os.path.exists(accuracy_path) and os.path.exists(loss_path)):
        return None

    accuracy_image = plt.imread(accuracy_path)
    loss_image = plt.imread(loss_path)

    figure, axes = plt.subplots(2, 1, figsize=(10, 12))
    axes[0].imshow(accuracy_image)
    axes[0].axis("off")
    axes[0].set_title("Accuracy Curve")
    axes[1].imshow(loss_image)
    axes[1].axis("off")
    axes[1].set_title("Loss Curve")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def rebuild_confusion_matrix(results_dir, config, best_config):
    try:
        import torch

        from data_loader import build_data_bundle
        from metrics import build_confusion_matrix, evaluate_model, normalize_confusion_matrix
        from model import HybridCNN
        from plotting import plot_confusion_matrix
    except ModuleNotFoundError:
        return None, None

    best_model_path = os.path.join(results_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        return None, None

    bundle = build_data_bundle(
        dataset=config.get("dataset", "cifar10"),
        data_dir=config.get("data_dir", "data"),
        batch_size=int(best_config.get("batch_size") or config.get("batch_size", 128)),
        train_size=int(config.get("train_size", 45000)),
        val_size=int(config.get("val_size", 5000)),
        seed=int(config.get("random_seed", 42)),
        num_workers=int(config.get("num_workers", 0)),
        subset_size=config.get("subset_size"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCNN(
        in_channels=bundle.in_channels,
        img_size=bundle.img_size,
        shared_conv_kernel_size=int(best_config["shared_conv_kernel_size"]),
        base_filters=int(best_config["base_filters"]),
        dilation=int(best_config["dilation"]),
        final_neurons=int(best_config["final_neurons"]),
        dropout=float(best_config["dropout"]),
        se_ratio=int(best_config["se_ratio"]),
        num_classes=bundle.num_classes,
    ).to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    test_metrics = evaluate_model(model, bundle.test_loader, device)
    confusion_matrix = build_confusion_matrix(test_metrics["targets"], test_metrics["predictions"], bundle.num_classes)
    confusion_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(confusion_matrix, CIFAR10_CLASS_NAMES, confusion_path)
    return confusion_matrix, confusion_path


def build_root_summary(results_dir, runs, config):
    if not runs:
        raise FileNotFoundError(f"No completed run artifacts were found in {results_dir}")

    best_run = max(runs, key=lambda item: item["gwo_test_accuracy"])
    aggregate = aggregate_stats(runs)
    timing = timing_stats(runs, sum(run["total_optimization_time"] + run["total_training_time"] for run in runs))
    caches = cache_stats(runs)

    summary_rows = []
    for index, run in enumerate(runs, start=1):
        summary_rows.append(
            {
                "run_id": index,
                "baseline_acc": f"{run['baseline_test_accuracy']:.6f}",
                "gwo_acc": f"{run['gwo_test_accuracy']:.6f}",
                "improvement": f"{run['accuracy_improvement_points']:.6f}",
                "best_shared_conv_kernel_size": run["best_config"].get("shared_conv_kernel_size"),
                "best_base_filters": run["best_config"].get("base_filters"),
                "best_dilation": run["best_config"].get("dilation"),
                "best_final_neurons": run["best_config"].get("final_neurons"),
                "best_dropout": f"{float(run['best_config'].get('dropout', 0.0)):.4f}",
                "best_learning_rate": f"{float(run['best_config'].get('learning_rate', 0.0)):.6f}",
                "best_batch_size": run["best_config"].get("batch_size"),
                "best_se_ratio": run["best_config"].get("se_ratio"),
            }
        )

    write_csv(
        os.path.join(results_dir, "summary.csv"),
        [
            "run_id",
            "baseline_acc",
            "gwo_acc",
            "improvement",
            "best_shared_conv_kernel_size",
            "best_base_filters",
            "best_dilation",
            "best_final_neurons",
            "best_dropout",
            "best_learning_rate",
            "best_batch_size",
            "best_se_ratio",
        ],
        summary_rows,
    )

    write_statistics_file(os.path.join(results_dir, "statistics.txt"), aggregate)
    write_timing_file(os.path.join(results_dir, "timing.txt"), timing)
    write_cache_statistics_file(os.path.join(results_dir, "cache_statistics.txt"), caches)

    search_rows = []
    for run in runs:
        search_history_path = run["search_history_path"]
        search_rows.extend(read_search_history(search_history_path))

    iteration_summaries = build_iteration_summaries(search_rows)
    unique_solution_keys = set()
    for row in search_rows:
        unique_solution_keys.add(
            row.get("combination_key")
            or "|".join(
                [
                    str(row.get("shared_conv_kernel_size", "")),
                    str(row.get("base_filters", "")),
                    str(row.get("dilation", "")),
                    str(row.get("final_neurons", "")),
                    str(row.get("dropout", "")),
                    str(row.get("learning_rate", "")),
                    str(row.get("batch_size", "")),
                    str(row.get("se_ratio", "")),
                ]
            )
        )

    write_convergence_file(
        os.path.join(results_dir, "convergence.txt"),
        iteration_summaries,
        unique_solutions_evaluated=len(unique_solution_keys),
    )
    write_diversity_file(os.path.join(results_dir, "diversity_analysis.txt"), iteration_summaries)

    rebuild_training_curves(results_dir)
    confusion_matrix, confusion_path = rebuild_confusion_matrix(results_dir, config, best_run["best_config"])
    if confusion_matrix is not None:
        confusion_analysis = analyze_confusion_matrix(confusion_matrix, CIFAR10_CLASS_NAMES)
        write_confusion_analysis_file(os.path.join(results_dir, "confusion_analysis.txt"), confusion_analysis)

    root_final_results_path = os.path.join(results_dir, "final_results.txt")
    accuracy_improvement_points = (best_run["gwo_test_accuracy"] - best_run["baseline_test_accuracy"]) * 100.0
    relative_improvement = ((best_run["gwo_test_accuracy"] - best_run["baseline_test_accuracy"]) / best_run["baseline_test_accuracy"] * 100.0) if best_run["baseline_test_accuracy"] > 0 else 0.0
    total_fitness_evaluations = sum(run["total_evaluations"] for run in runs)
    total_training_time = sum(run["total_training_time"] for run in runs)

    lines = [
        f"Population Size: {config.get('population_size', 0)}",
        f"Iteration Count: {config.get('iteration_count', 0)}",
        f"Runs: {config.get('runs', len(runs))}",
        f"Search Epochs: {config.get('search_epochs', 0)}",
        f"Final Epochs: {config.get('final_epochs', 0)}",
        f"Batch Size: {config.get('batch_size', 0)}",
        f"Learning Rate: {config.get('learning_rate', 0.0)}",
        f"Random Seed: {config.get('random_seed', 0)}",
        f"Total Evaluations: {total_fitness_evaluations}",
        "",
        f"Unique Solutions: {best_run['unique_solutions']}",
        f"Cache Hits: {caches['cache_hits']}",
        f"Cache Misses: {caches['cache_misses']}",
        f"Cache Reuse Rate (%): {caches['cache_reuse_rate']:.2f}",
        "",
        f"Baseline Test Accuracy: {best_run['baseline_test_accuracy']:.6f}",
        f"GWO Test Accuracy: {best_run['gwo_test_accuracy']:.6f}",
        f"Accuracy Improvement (percentage points): {accuracy_improvement_points:.2f}",
        f"Relative Improvement (%): {relative_improvement:.6f}",
        "",
        "Best Hyperparameters:",
    ]
    for key, value in best_run["best_config"].items():
        lines.append(f"  {key}: {value}")
    lines.extend(
        [
            "",
            f"Total Optimization Time (seconds): {best_run['total_optimization_time']:.2f}",
            f"Total Training Time (seconds): {total_training_time:.2f}",
            f"Total Experiment Time (seconds): {total_training_time:.2f}",
        ]
    )
    write_text(root_final_results_path, "\n".join(lines) + "\n")

    return {
        "summary_rows": summary_rows,
        "best_run": best_run,
        "statistics_path": os.path.join(results_dir, "statistics.txt"),
        "timing_path": os.path.join(results_dir, "timing.txt"),
        "cache_statistics_path": os.path.join(results_dir, "cache_statistics.txt"),
        "convergence_path": os.path.join(results_dir, "convergence.txt"),
        "diversity_path": os.path.join(results_dir, "diversity_analysis.txt"),
        "training_curves_path": os.path.join(results_dir, "training_curves.png"),
        "confusion_matrix_path": confusion_path,
        "final_results_path": root_final_results_path,
    }


def load_runs(results_dir):
    runs = []
    for run_dir in find_run_dirs(results_dir):
        final_results_path = os.path.join(run_dir, "final_results.txt")
        best_config_path = os.path.join(run_dir, "best_config.txt")
        config_path = os.path.join(run_dir, "config_used.json")
        search_history_path = os.path.join(run_dir, "search_history.csv")

        if not os.path.exists(final_results_path) or not os.path.exists(best_config_path):
            continue

        final_results = parse_final_results(final_results_path)
        best_config = parse_best_config(best_config_path)
        config = load_config(config_path)

        runs.append(
            {
                **final_results,
                "best_config": best_config,
                "config": config,
                "search_history_path": search_history_path,
            }
        )
    return runs


def main():
    parser = argparse.ArgumentParser(description="Rebuild experiment summary artifacts without rerunning training.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory that already contains the completed run artifacts.")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    runs = load_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"No completed run artifacts found in {results_dir}")

    config = runs[0].get("config", {})
    summary = build_root_summary(results_dir, runs, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
