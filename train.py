import argparse
import copy
import csv
import os
import shutil
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split

from config import config_to_dict, load_config, save_config
from data_loader import DataBundle, build_data_bundle, build_data_loader
from gwo import GreyWolfOptimizer
from metrics import build_confusion_matrix, evaluate_model, normalize_confusion_matrix, train_one_epoch
from model import HybridCNN
from plotting import plot_confusion_matrix, plot_curves, plot_global, plot_locals
from utils import ensure_dir, make_torch_generator, set_global_seed, setup_logging, write_json

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def build_model(model_config, in_channels, img_size, num_classes):
    return HybridCNN(
        in_channels=in_channels,
        img_size=img_size,
        kernel_size=model_config["kernel_size"],
        base_filters=model_config["base_filters"],
        dilation=model_config["dilation"],
        final_neurons=model_config["final_neurons"],
        dropout=model_config["dropout"],
        se_ratio=model_config["se_ratio"],
        num_classes=num_classes,
    )


def map_position_to_config(position, config):
    search_space = config.search_space
    kernel_sizes = search_space["kernel_sizes"]
    base_filters = search_space["base_filters"]
    dilations = search_space["dilations"]
    final_neurons = search_space["final_neurons"]
    batch_sizes = search_space["batch_sizes"]
    se_ratios = search_space["se_ratios"]

    kernel_idx = int(round(position[0]))
    base_idx = int(round(position[1]))
    dilation_idx = int(round(position[2]))
    final_idx = int(round(position[3]))
    dropout = float(position[4])
    learning_rate_log = float(position[5])
    batch_idx = int(round(position[6]))
    se_idx = int(round(position[7]))

    kernel_idx = max(0, min(kernel_idx, len(kernel_sizes) - 1))
    base_idx = max(0, min(base_idx, len(base_filters) - 1))
    dilation_idx = max(0, min(dilation_idx, len(dilations) - 1))
    final_idx = max(0, min(final_idx, len(final_neurons) - 1))
    batch_idx = max(0, min(batch_idx, len(batch_sizes) - 1))
    se_idx = max(0, min(se_idx, len(se_ratios) - 1))

    learning_rate_min = search_space["learning_rate_min"]
    learning_rate_max = search_space["learning_rate_max"]
    learning_rate_log_min = float(np.log10(learning_rate_min))
    learning_rate_log_max = float(np.log10(learning_rate_max))
    learning_rate_log = max(learning_rate_log_min, min(learning_rate_log, learning_rate_log_max))
    learning_rate = float(10**learning_rate_log)

    return {
        "kernel_size": kernel_sizes[kernel_idx],
        "base_filters": base_filters[base_idx],
        "dilation": dilations[dilation_idx],
        "final_neurons": final_neurons[final_idx],
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_sizes[batch_idx],
        "se_ratio": se_ratios[se_idx],
    }


def build_bounds(config):
    search_space = config.search_space
    return [
        (0, len(search_space["kernel_sizes"]) - 1),
        (0, len(search_space["base_filters"]) - 1),
        (0, len(search_space["dilations"]) - 1),
        (0, len(search_space["final_neurons"]) - 1),
        (search_space["dropout_min"], search_space["dropout_max"]),
        (float(np.log10(search_space["learning_rate_min"])), float(np.log10(search_space["learning_rate_max"]))),
        (0, len(search_space["batch_sizes"]) - 1),
        (0, len(search_space["se_ratios"]) - 1),
    ]


def position_signature(position, config):
    model_config = map_position_to_config(position, config)
    return (
        model_config["kernel_size"],
        model_config["base_filters"],
        model_config["dilation"],
        model_config["final_neurons"],
        round(model_config["dropout"], 4),
        round(model_config["learning_rate"], 6),
        model_config["batch_size"],
        model_config["se_ratio"],
    )


def solution_cache_key(model_config):
    return (
        model_config["kernel_size"],
        model_config["base_filters"],
        model_config["dilation"],
        model_config["final_neurons"],
        round(model_config["dropout"], 4),
        round(model_config["learning_rate"], 6),
        model_config["batch_size"],
        model_config["se_ratio"],
    )


def build_candidate_loaders(bundle: DataBundle, batch_size: int, seed: int, num_workers: int):
    train_loader = build_data_loader(bundle.train_dataset, batch_size, seed, num_workers, shuffle=True)
    val_loader = build_data_loader(bundle.val_dataset, batch_size, seed + 1, num_workers, shuffle=False)
    return train_loader, val_loader


def create_optimizer(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    learning_rate,
    epochs,
    patience,
    logger,
    checkpoint_path=None,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, learning_rate)

    history = {
        "train_accuracy": [],
        "val_accuracy": [],
        "train_loss": [],
        "val_loss": [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -1.0
    best_epoch = 0
    patience_counter = 0

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_metrics = evaluate_model(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        logger.info(
            "epoch=%s train_loss=%.6f train_acc=%.6f val_loss=%.6f val_acc=%.6f",
            epoch,
            train_loss,
            train_accuracy,
            val_metrics["loss"],
            val_metrics["accuracy"],
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path is not None:
                ensure_dir(os.path.dirname(checkpoint_path))
                torch.save(best_state, checkpoint_path)
        else:
            patience_counter += 1

        if patience is not None and patience_counter >= patience:
            logger.info("early_stopping triggered at epoch=%s", epoch)
            break

    train_time = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    return model, history, best_val_accuracy, best_epoch, train_time


def append_search_history(csv_path, iteration, wolf_id, model_config, fitness, evaluation_time, cache_hit):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        if not file_exists:
            writer.writerow(["iteration", "wolf_id", "kernel_size", "base_filters", "dilation", "final_neurons", "dropout", "learning_rate", "batch_size", "se_ratio", "fitness", "evaluation_time", "cache_hit"])
        writer.writerow(
            [
                iteration,
                wolf_id,
                model_config["kernel_size"],
                model_config["base_filters"],
                model_config["dilation"],
                model_config["final_neurons"],
                f"{model_config['dropout']:.4f}",
                f"{model_config['learning_rate']:.6f}",
                model_config["batch_size"],
                model_config["se_ratio"],
                f"{fitness:.6f}",
                f"{evaluation_time:.6f}",
                cache_hit,
            ]
        )


def write_search_space_file(config, output_path):
    search_space = config.search_space
    payload = {
        "KERNEL_SIZES": search_space["kernel_sizes"],
        "BASE_FILTERS": search_space["base_filters"],
        "DILATIONS": search_space["dilations"],
        "FINAL_NEURONS": search_space["final_neurons"],
        "SE_RATIOS": search_space["se_ratios"],
        "BATCH_SIZES": search_space["batch_sizes"],
        "LEARNING_RATE_MIN": search_space["learning_rate_min"],
        "LEARNING_RATE_MAX": search_space["learning_rate_max"],
        "DROPOUT_MIN": search_space["dropout_min"],
        "DROPOUT_MAX": search_space["dropout_max"],
    }
    write_json(output_path, payload)


def augment_search_history_with_diversity(csv_path, iteration_summaries):
    if not os.path.exists(csv_path):
        return

    with open(csv_path, mode="r", newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
        fieldnames = list(rows[0].keys()) if rows else []

    summary_by_iteration = {summary["iteration"]: summary for summary in iteration_summaries}
    for row in rows:
        iteration = int(row["iteration"])
        summary = summary_by_iteration.get(iteration, {})
        row["combination_key"] = f"ks={row['kernel_size']};bf={row['base_filters']};d={row['dilation']};fn={row['final_neurons']};dr={row['dropout']};lr={row.get('learning_rate', '')};bs={row.get('batch_size', '')};se={row.get('se_ratio', '')}"
        row["iteration_unique_solutions"] = summary.get("unique_solution_count", "")
        row["iteration_repeat_rate"] = f"{summary.get('repeat_rate', 0.0):.6f}" if summary else ""
        row["iteration_avg_fitness"] = f"{summary.get('average_fitness', 0.0):.6f}" if summary else ""
        row["iteration_best_fitness"] = f"{summary.get('best_fitness', 0.0):.6f}" if summary else ""

    extended_fieldnames = fieldnames + [
        "combination_key",
        "iteration_unique_solutions",
        "iteration_repeat_rate",
        "iteration_avg_fitness",
        "iteration_best_fitness",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=extended_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_convergence_metrics(global_bests):
    if not global_bests:
        return {
            "first_fitness": 0.0,
            "last_fitness": 0.0,
            "fitness_increase": 0.0,
            "convergence_percent": 0.0,
        }

    first_fitness = float(global_bests[0])
    last_fitness = float(global_bests[-1])
    increase = last_fitness - first_fitness
    convergence_percent = (increase / abs(first_fitness) * 100.0) if first_fitness != 0 else 0.0
    return {
        "first_fitness": first_fitness,
        "last_fitness": last_fitness,
        "fitness_increase": increase,
        "convergence_percent": convergence_percent,
    }


def analyze_confusion_matrix(confusion_matrix, class_names):
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


def write_convergence_file(output_path, convergence_metrics):
    with open(output_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("iteration,best_fitness,average_fitness,unique_solution_count\n")
        for summary in convergence_metrics.get("iteration_summaries", []):
            file_handle.write(f"{summary['iteration']},{summary['best_fitness']:.6f},{summary['average_fitness']:.6f},{summary['unique_solution_count']}\n")
        file_handle.write("\n")
        file_handle.write(f"Unique Solutions Evaluated: {convergence_metrics.get('unique_solutions_evaluated', 0)}\n")
        file_handle.write(f"first_iter_fitness: {convergence_metrics['first_fitness']:.6f}\n")
        file_handle.write(f"last_iter_fitness: {convergence_metrics['last_fitness']:.6f}\n")
        file_handle.write(f"fitness_increase: {convergence_metrics['fitness_increase']:.6f}\n")
        file_handle.write(f"convergence_percent: {convergence_metrics['convergence_percent']:.2f}\n")


def write_confusion_analysis_file(output_path, confusion_analysis):
    with open(output_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("Top 5 Confused Class Pairs:\n")
        for count, true_name, pred_name in confusion_analysis["top_confusions"]:
            file_handle.write(f"{true_name} -> {pred_name}: {count}\n")
        file_handle.write(f"Best Accuracy Class: {confusion_analysis['best_class'][0]} ({confusion_analysis['best_class'][1]:.6f})\n")
        file_handle.write(f"Worst Accuracy Class: {confusion_analysis['worst_class'][0]} ({confusion_analysis['worst_class'][1]:.6f})\n")


def write_diversity_analysis_file(output_path, iteration_summaries):
    with open(output_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("iteration,unique_solution_count,repeat_rate,average_fitness,best_fitness\n")
        for summary in iteration_summaries:
            file_handle.write(f"{summary['iteration']},{summary['unique_solution_count']},{summary['repeat_rate']:.6f},{summary['average_fitness']:.6f},{summary['best_fitness']:.6f}\n")


def write_timing_file(output_path, timing_info):
    with open(output_path, "w", encoding="utf-8") as file_handle:
        for key, value in timing_info.items():
            if isinstance(value, dict):
                file_handle.write(f"{key}: mean={value['mean']:.2f}, std={value['std']:.2f}\n")
            else:
                file_handle.write(f"{key}: {value:.2f}\n")


def write_statistics_file(output_path, stats):
    with open(output_path, "w", encoding="utf-8") as file_handle:
        for key, value in stats.items():
            if isinstance(value, dict):
                file_handle.write(f"{key}: mean={value['mean']:.6f}, std={value['std']:.6f}\n")
            else:
                file_handle.write(f"{key}: {value:.6f}\n")


def write_cache_statistics_file(output_path, cache_stats):
    with open(output_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"Total Evaluations: {cache_stats['total_evaluations']}\n")
        file_handle.write(f"Real Training Count: {cache_stats['real_training_count']}\n")
        file_handle.write(f"Cache Hits: {cache_stats['cache_hits']}\n")
        file_handle.write(f"Cache Misses: {cache_stats['cache_misses']}\n")
        file_handle.write(f"Saved Training Count: {cache_stats['saved_training_count']}\n")
        file_handle.write(f"Saved Estimated Time (seconds): {cache_stats['saved_estimated_time']:.6f}\n")
        file_handle.write(f"Cache Reuse Rate (%): {cache_stats['cache_reuse_rate']:.2f}\n")


def write_summary_csv(output_path, rows):
    fieldnames = [
        "run_id",
        "baseline_acc",
        "gwo_acc",
        "improvement",
        "best_kernel_size",
        "best_base_filters",
        "best_dilation",
        "best_final_neurons",
        "best_dropout",
        "best_learning_rate",
        "best_batch_size",
        "best_se_ratio",
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_search_history_aggregate(output_path, runs_results):
    fieldnames = [
        "run_id",
        "iteration",
        "wolf_id",
        "kernel_size",
        "base_filters",
        "dilation",
        "final_neurons",
        "dropout",
        "learning_rate",
        "batch_size",
        "se_ratio",
        "fitness",
        "evaluation_time",
        "cache_hit",
        "combination_key",
        "iteration_unique_solutions",
        "iteration_repeat_rate",
        "iteration_avg_fitness",
        "iteration_best_fitness",
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for run_index, run_result in enumerate(runs_results, start=1):
            search_history_path = os.path.join(run_result["run_dir"], "search_history.csv")
            if not os.path.exists(search_history_path):
                continue
            with open(search_history_path, mode="r", newline="", encoding="utf-8") as input_file:
                reader = csv.DictReader(input_file)
                for row in reader:
                    row["run_id"] = run_index
                    writer.writerow(row)


def compute_aggregate_stats(runs_results):
    baseline_test = np.array([run["baseline_result"]["test_accuracy"] for run in runs_results], dtype=float)
    gwo_test = np.array([run["final_result"]["test_accuracy"] for run in runs_results], dtype=float)
    improvements = np.array([run["improvement"] for run in runs_results], dtype=float)
    return {
        "Average Accuracy": {"mean": float(np.mean(gwo_test)), "std": float(np.std(gwo_test, ddof=0))},
        "Baseline Accuracy": {"mean": float(np.mean(baseline_test)), "std": float(np.std(baseline_test, ddof=0))},
        "Average Improvement": {"mean": float(np.mean(improvements)), "std": float(np.std(improvements, ddof=0))},
    }


def compute_timing_aggregate(runs_results, total_experiment_time):
    optimization_times = np.array([run["search_result"]["optimization_time"] for run in runs_results], dtype=float)
    baseline_times = np.array([run["baseline_result"]["training_time"] for run in runs_results], dtype=float)
    final_times = np.array([run["final_result"]["training_time"] for run in runs_results], dtype=float)
    return {
        "GWO Optimization Time": {"mean": float(np.mean(optimization_times)), "std": float(np.std(optimization_times, ddof=0))},
        "Baseline Training Time": {"mean": float(np.mean(baseline_times)), "std": float(np.std(baseline_times, ddof=0))},
        "Final GWO Training Time": {"mean": float(np.mean(final_times)), "std": float(np.std(final_times, ddof=0))},
        "Total Experiment Time": float(total_experiment_time),
    }


def compute_cache_aggregate(runs_results):
    cache_hits = sum(run["search_result"].get("cache_hits", 0) for run in runs_results)
    cache_misses = sum(run["search_result"].get("cache_misses", 0) for run in runs_results)
    saved_training_count = sum(run["search_result"].get("saved_training_count", 0) for run in runs_results)
    saved_estimated_time = sum(run["search_result"].get("saved_estimated_time", 0.0) for run in runs_results)
    total_evaluations = cache_hits + cache_misses
    real_training_count = cache_misses
    return {
        "total_evaluations": int(total_evaluations),
        "real_training_count": int(real_training_count),
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "saved_training_count": int(saved_training_count),
        "saved_estimated_time": float(saved_estimated_time),
        "cache_reuse_rate": (cache_hits / max(total_evaluations, 1)) * 100.0,
    }


def select_best_run(runs_results):
    return max(runs_results, key=lambda run: run["final_result"]["test_accuracy"])


def build_root_artifacts(base_config, runs_results, total_experiment_time):
    best_run = select_best_run(runs_results)
    root_dir = base_config.results_dir

    write_search_space_file(base_config, os.path.join(root_dir, "search_space.json"))
    write_search_history_aggregate(os.path.join(root_dir, "search_history.csv"), runs_results)

    summary_rows = []
    for run_index, run_result in enumerate(runs_results, start=1):
        baseline_acc = run_result["baseline_result"]["test_accuracy"]
        gwo_acc = run_result["final_result"]["test_accuracy"]
        improvement = run_result["improvement"]
        summary_rows.append(
            {
                "run_id": run_index,
                "baseline_acc": f"{baseline_acc:.6f}",
                "gwo_acc": f"{gwo_acc:.6f}",
                "improvement": f"{improvement:.6f}",
                "best_kernel_size": run_result["best_config"]["kernel_size"],
                "best_base_filters": run_result["best_config"]["base_filters"],
                "best_dilation": run_result["best_config"]["dilation"],
                "best_final_neurons": run_result["best_config"]["final_neurons"],
                "best_dropout": f"{run_result['best_config']['dropout']:.4f}",
                "best_learning_rate": f"{run_result['best_config']['learning_rate']:.6f}",
                "best_batch_size": run_result["best_config"]["batch_size"],
                "best_se_ratio": run_result["best_config"]["se_ratio"],
            }
        )
    write_summary_csv(os.path.join(root_dir, "summary.csv"), summary_rows)

    aggregate_stats = compute_aggregate_stats(runs_results)
    write_statistics_file(os.path.join(root_dir, "statistics.txt"), aggregate_stats)

    timing_info = compute_timing_aggregate(runs_results, total_experiment_time)
    write_timing_file(os.path.join(root_dir, "timing.txt"), timing_info)

    cache_stats = compute_cache_aggregate(runs_results)
    write_cache_statistics_file(os.path.join(root_dir, "cache_statistics.txt"), cache_stats)

    convergence_metrics = compute_convergence_metrics(best_run["search_result"]["global_bests"])
    convergence_metrics["iteration_summaries"] = best_run["search_result"]["iteration_summaries"]
    convergence_metrics["unique_solutions_evaluated"] = best_run["search_result"].get("unique_solutions_evaluated", 0)
    write_convergence_file(os.path.join(root_dir, "convergence.txt"), convergence_metrics)

    confusion_analysis = analyze_confusion_matrix(best_run["final_result"]["confusion_matrix"], CLASS_NAMES)
    write_confusion_analysis_file(os.path.join(root_dir, "confusion_analysis.txt"), confusion_analysis)
    write_diversity_analysis_file(os.path.join(root_dir, "diversity_analysis.txt"), best_run["search_result"]["iteration_summaries"])

    best_config_path = os.path.join(root_dir, "best_config.txt")
    with open(best_config_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("Best Hyperparameters:\n")
        for key, value in best_run["best_config"].items():
            file_handle.write(f"{key}: {value}\n")
        file_handle.write(f"validation_accuracy: {best_run['search_result']['best_fitness']:.6f}\n")
        file_handle.write(f"learning_rate: {best_run['best_config']['learning_rate']:.6f}\n")
        file_handle.write(f"batch_size: {best_run['best_config']['batch_size']}\n")

    final_results_path = os.path.join(root_dir, "final_results.txt")
    relative_improvement = ((best_run["final_result"]["test_accuracy"] - best_run["baseline_result"]["test_accuracy"]) / best_run["baseline_result"]["test_accuracy"] * 100.0) if best_run["baseline_result"]["test_accuracy"] > 0 else 0.0
    accuracy_improvement = best_run["final_result"]["test_accuracy"] - best_run["baseline_result"]["test_accuracy"]
    total_fitness_evaluations = sum(run["search_result"]["evaluation_count"] for run in runs_results)

    with open(final_results_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"Population Size: {base_config.population_size}\n")
        file_handle.write(f"Iteration Count: {base_config.iteration_count}\n")
        file_handle.write(f"Runs: {base_config.runs}\n")
        file_handle.write(f"Search Epochs: {base_config.search_epochs}\n")
        file_handle.write(f"Final Epochs: {base_config.final_epochs}\n")
        file_handle.write(f"Batch Size: {base_config.batch_size}\n")
        file_handle.write(f"Learning Rate: {base_config.learning_rate}\n")
        file_handle.write(f"Random Seed: {base_config.random_seed}\n")
        file_handle.write(f"Total Fitness Evaluations: {total_fitness_evaluations}\n\n")
        file_handle.write(f"Cache Hits: {cache_stats['cache_hits']}\n")
        file_handle.write(f"Cache Misses: {cache_stats['cache_misses']}\n")
        file_handle.write(f"Cache Reuse Rate (%): {cache_stats['cache_reuse_rate']:.2f}\n\n")
        file_handle.write("Baseline Validation Accuracy: {0:.6f}\n".format(best_run["baseline_result"]["validation_accuracy"]))
        file_handle.write("Baseline Test Accuracy: {0:.6f}\n".format(best_run["baseline_result"]["test_accuracy"]))
        file_handle.write("GWO Validation Accuracy: {0:.6f}\n".format(best_run["search_result"]["best_fitness"]))
        file_handle.write("GWO Test Accuracy: {0:.6f}\n".format(best_run["final_result"]["test_accuracy"]))
        file_handle.write(f"Accuracy Improvement (% points): {accuracy_improvement:.6f}\n")
        file_handle.write(f"Relative Improvement (%): {relative_improvement:.6f}\n\n")
        file_handle.write("Best Hyperparameters:\n")
        for key, value in best_run["best_config"].items():
            file_handle.write(f"  {key}: {value}\n")
        file_handle.write("\n")
        file_handle.write(f"Best Learning Rate: {best_run['best_config']['learning_rate']:.6f}\n")
        file_handle.write(f"Best Batch Size: {best_run['best_config']['batch_size']}\n")
        file_handle.write(f"GWO Optimization Time (seconds): {best_run['search_result']['optimization_time']:.2f}\n")
        file_handle.write(f"Baseline Training Time (seconds): {best_run['baseline_result']['training_time']:.2f}\n")
        file_handle.write(f"Final GWO Training Time (seconds): {best_run['final_result']['training_time']:.2f}\n")
        file_handle.write(f"Total Experiment Time (seconds): {total_experiment_time:.2f}\n")

    artifact_map = {
        "best_model.pth": os.path.join(os.path.dirname(best_run["artifacts"]["best_config_path"]), "best_model.pth"),
        "best_config.txt": best_config_path,
        "final_results.txt": final_results_path,
        "statistics.txt": os.path.join(root_dir, "statistics.txt"),
        "timing.txt": os.path.join(root_dir, "timing.txt"),
        "cache_statistics.txt": os.path.join(root_dir, "cache_statistics.txt"),
        "convergence.txt": os.path.join(root_dir, "convergence.txt"),
        "confusion_analysis.txt": os.path.join(root_dir, "confusion_analysis.txt"),
        "summary.csv": os.path.join(root_dir, "summary.csv"),
        "search_history.csv": os.path.join(root_dir, "search_history.csv"),
        "search_space.json": os.path.join(root_dir, "search_space.json"),
        "config_used.json": os.path.join(root_dir, "config_used.json"),
        "global_best.png": os.path.join(os.path.dirname(best_run["artifacts"]["best_config_path"]), "global_best.png"),
        "local_bests.png": os.path.join(os.path.dirname(best_run["artifacts"]["best_config_path"]), "local_bests.png"),
        "accuracy_curve.png": os.path.join(os.path.dirname(best_run["artifacts"]["best_config_path"]), "accuracy_curve.png"),
        "loss_curve.png": os.path.join(os.path.dirname(best_run["artifacts"]["best_config_path"]), "loss_curve.png"),
        "confusion_matrix.png": os.path.join(os.path.dirname(best_run["artifacts"]["best_config_path"]), "confusion_matrix.png"),
        "run.log": os.path.join(root_dir, "run.log"),
    }

    for filename, source_path in artifact_map.items():
        if os.path.exists(source_path):
            target_path = os.path.join(root_dir, filename)
            if os.path.abspath(source_path) != os.path.abspath(target_path):
                shutil.copy2(source_path, target_path)

    return {
        "best_run": best_run,
        "aggregate_stats": aggregate_stats,
        "timing_info": timing_info,
        "convergence_metrics": convergence_metrics,
        "confusion_analysis": confusion_analysis,
        "summary_rows": summary_rows,
    }


def build_final_training_bundle(bundle: DataBundle, batch_size, seed, num_workers):
    combined_dataset = ConcatDataset([bundle.train_dataset, bundle.val_dataset])
    monitor_size = max(1, int(len(combined_dataset) * 0.1))
    train_size = len(combined_dataset) - monitor_size
    split_generator = make_torch_generator(seed + 101)
    final_train_subset, monitor_subset = random_split(combined_dataset, [train_size, monitor_size], generator=split_generator)

    loader_generator = make_torch_generator(seed + 102)

    def seed_worker(worker_id):
        torch.manual_seed(seed + 102 + worker_id)
        np.random.seed(seed + 102 + worker_id)
        random.seed(seed + 102 + worker_id)

    final_train_loader = DataLoader(
        final_train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
        worker_init_fn=seed_worker,
    )
    monitor_loader = DataLoader(
        monitor_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return final_train_loader, monitor_loader


def run_baseline_experiment(bundle, config, device, logger, run_dir):
    baseline_config = {
        "kernel_size": 3,
        "base_filters": 32,
        "dilation": 1,
        "final_neurons": 256,
        "dropout": 0.25,
        "se_ratio": 8,
    }
    model = build_model(baseline_config, bundle.in_channels, bundle.img_size, bundle.num_classes).to(device)
    checkpoint_path = os.path.join(run_dir, "baseline_best.pth")
    model, baseline_history, baseline_val_accuracy, _, baseline_training_time = train_model(
        model,
        bundle.train_loader,
        bundle.val_loader,
        device,
        learning_rate=config.learning_rate,
        epochs=config.final_epochs,
        patience=config.patience,
        logger=logger,
        checkpoint_path=checkpoint_path,
    )
    baseline_test_metrics = evaluate_model(model, bundle.test_loader, device)
    return {
        "config": baseline_config,
        "history": baseline_history,
        "validation_accuracy": baseline_val_accuracy,
        "test_accuracy": baseline_test_metrics["accuracy"],
        "test_loss": baseline_test_metrics["loss"],
        "training_time": baseline_training_time,
        "checkpoint_path": checkpoint_path,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }


def run_gwo_search(bundle, config, device, logger, run_dir):
    search_history_path = os.path.join(run_dir, "search_history.csv")
    if os.path.exists(search_history_path):
        os.remove(search_history_path)

    bounds = build_bounds(config)
    search_start = time.perf_counter()
    fitness_cache = {}
    cache_hits = 0
    cache_misses = 0
    saved_estimated_time = 0.0

    def fitness(position, iteration=None, wolf_id=None):
        nonlocal cache_hits, cache_misses, saved_estimated_time
        model_config = map_position_to_config(position, config)
        solution_key = solution_cache_key(model_config)

        if solution_key in fitness_cache:
            cached_entry = fitness_cache[solution_key]
            val_accuracy = cached_entry["fitness"]
            evaluation_time = 0.0
            cache_hit = True
            cache_hits += 1
            saved_estimated_time += cached_entry["evaluation_time"]
        else:
            eval_seed = config.random_seed + (iteration or 0) * 100 + (wolf_id or 0)
            train_loader, val_loader = build_candidate_loaders(bundle, model_config["batch_size"], eval_seed, config.num_workers)
            model = build_model(model_config, bundle.in_channels, bundle.img_size, bundle.num_classes).to(device)
            eval_start = time.perf_counter()
            _, history, val_accuracy, _, _ = train_model(
                model,
                train_loader,
                val_loader,
                device,
                learning_rate=model_config["learning_rate"],
                epochs=config.search_epochs,
                patience=config.patience,
                logger=logger,
            )
            evaluation_time = time.perf_counter() - eval_start
            fitness_cache[solution_key] = {
                "fitness": val_accuracy,
                "evaluation_time": evaluation_time,
            }
            cache_hit = False
            cache_misses += 1

        append_search_history(search_history_path, iteration, wolf_id, model_config, val_accuracy, evaluation_time, cache_hit)
        logger.info(
            "gwo_iteration=%s wolf_id=%s val_acc=%.6f cache_hit=%s config=%s",
            iteration,
            wolf_id,
            val_accuracy,
            cache_hit,
            model_config,
        )
        return val_accuracy

    gwo = GreyWolfOptimizer(
        fitness,
        bounds,
        population=config.population_size,
        iterations=config.iteration_count,
        solution_signature_func=lambda position: position_signature(position, config),
    )
    result = gwo.optimize()
    augment_search_history_with_diversity(search_history_path, result["iteration_summaries"])
    search_time = time.perf_counter() - search_start
    result["optimization_time"] = search_time
    result["search_history_path"] = search_history_path
    result["cache_hits"] = cache_hits
    result["cache_misses"] = cache_misses
    result["cache_reuse_rate"] = (cache_hits / max(cache_hits + cache_misses, 1)) * 100.0
    result["unique_solutions_evaluated"] = len(fitness_cache)
    result["real_training_count"] = cache_misses
    result["saved_training_count"] = cache_hits
    result["saved_estimated_time"] = saved_estimated_time
    result["fitness_cache"] = fitness_cache
    return result


def run_final_gwo_experiment(bundle, config, best_config, device, logger, run_dir):
    final_train_loader, monitor_loader = build_final_training_bundle(bundle, best_config["batch_size"], config.random_seed, config.num_workers)
    model = build_model(best_config, bundle.in_channels, bundle.img_size, bundle.num_classes).to(device)
    best_model_path = os.path.join(run_dir, "best_model.pth")
    model, history, best_monitor_accuracy, _, training_time = train_model(
        model,
        final_train_loader,
        monitor_loader,
        device,
        learning_rate=best_config["learning_rate"],
        epochs=config.final_epochs,
        patience=config.patience,
        logger=logger,
        checkpoint_path=best_model_path,
    )
    test_metrics = evaluate_model(model, bundle.test_loader, device)
    confusion_matrix = build_confusion_matrix(test_metrics["targets"], test_metrics["predictions"], bundle.num_classes)
    return {
        "model": model,
        "history": history,
        "monitor_accuracy": best_monitor_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_metrics": test_metrics,
        "confusion_matrix": confusion_matrix,
        "training_time": training_time,
        "best_model_path": best_model_path,
        "final_train_loader": final_train_loader,
        "monitor_loader": monitor_loader,
    }


def save_run_summary(run_dir, best_config, search_result, baseline_result, final_result, config):
    best_config_path = os.path.join(run_dir, "best_config.txt")
    with open(best_config_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("Best Hyperparameters:\n")
        for key, value in best_config.items():
            file_handle.write(f"{key}: {value}\n")
        file_handle.write(f"validation_accuracy: {search_result['best_fitness']:.6f}\n")

    plot_global(search_result["global_bests"], os.path.join(run_dir, "global_best.png"))
    plot_locals(search_result["local_bests"], os.path.join(run_dir, "local_bests.png"))
    plot_curves(final_result["history"], os.path.join(run_dir, "accuracy_curve.png"), os.path.join(run_dir, "loss_curve.png"))

    confusion_matrix = build_confusion_matrix(final_result["test_metrics"]["targets"], final_result["test_metrics"]["predictions"], len(CLASS_NAMES))
    plot_confusion_matrix(confusion_matrix, CLASS_NAMES, os.path.join(run_dir, "confusion_matrix.png"))

    summary_path = os.path.join(run_dir, "final_results.txt")
    improvement = ((final_result["test_accuracy"] - baseline_result["test_accuracy"]) / baseline_result["test_accuracy"] * 100.0) if baseline_result["test_accuracy"] > 0 else 0.0

    with open(summary_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"population size: {config.population_size}\n")
        file_handle.write(f"iteration count: {config.iteration_count}\n")
        file_handle.write(f"total fitness evaluations: {search_result['evaluation_count']}\n")
        file_handle.write("best hyperparameters:\n")
        for key, value in best_config.items():
            file_handle.write(f"  {key}: {value}\n")
        file_handle.write(f"best validation accuracy: {search_result['best_fitness']:.6f}\n")
        file_handle.write(f"baseline validation accuracy: {baseline_result['validation_accuracy']:.6f}\n")
        file_handle.write(f"baseline test accuracy: {baseline_result['test_accuracy']:.6f}\n")
        file_handle.write(f"GWO test accuracy: {final_result['test_accuracy']:.6f}\n")
        file_handle.write(f"accuracy improvement (%): {improvement:.2f}\n")
        file_handle.write(f"training time (seconds): {final_result['training_time']:.2f}\n")
        file_handle.write(f"optimization time (seconds): {search_result['optimization_time']:.2f}\n")

    return {
        "best_config_path": best_config_path,
        "summary_path": summary_path,
        "accuracy_curve_path": os.path.join(run_dir, "accuracy_curve.png"),
        "loss_curve_path": os.path.join(run_dir, "loss_curve.png"),
        "global_best_path": os.path.join(run_dir, "global_best.png"),
        "local_bests_path": os.path.join(run_dir, "local_bests.png"),
        "confusion_matrix_path": os.path.join(run_dir, "confusion_matrix.png"),
    }


def execute_single_run(base_config, run_index, logger):
    run_seed = base_config.random_seed + run_index
    run_config = copy.deepcopy(base_config)
    run_config.random_seed = run_seed

    set_global_seed(run_seed)
    run_dir = base_config.results_dir if base_config.runs == 1 else os.path.join(base_config.results_dir, f"run_{run_index + 1:02d}")
    ensure_dir(run_dir)
    save_config(run_config, os.path.join(run_dir, "config_used.json"))

    bundle = build_data_bundle(
        dataset=run_config.dataset,
        data_dir=run_config.data_dir,
        batch_size=run_config.batch_size,
        train_size=run_config.train_size,
        val_size=run_config.val_size,
        seed=run_seed,
        num_workers=run_config.num_workers,
        subset_size=run_config.subset_size,
    )

    logger.info("run_start=%s seed=%s run_dir=%s", run_index + 1, run_seed, run_dir)

    search_result = run_gwo_search(bundle, run_config, torch.device("cuda" if torch.cuda.is_available() else "cpu"), logger, run_dir)
    best_config = map_position_to_config(search_result["best_pos"], run_config)
    baseline_result = run_baseline_experiment(bundle, run_config, torch.device("cuda" if torch.cuda.is_available() else "cpu"), logger, run_dir)
    final_result = run_final_gwo_experiment(bundle, run_config, best_config, torch.device("cuda" if torch.cuda.is_available() else "cpu"), logger, run_dir)
    final_result["test_metrics"] = evaluate_model(final_result["model"], bundle.test_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    final_result["test_accuracy"] = final_result["test_metrics"]["accuracy"]
    final_result["test_loss"] = final_result["test_metrics"]["loss"]
    improvement = (final_result["test_accuracy"] - baseline_result["test_accuracy"]) * 100.0
    relative_improvement = ((final_result["test_accuracy"] - baseline_result["test_accuracy"]) / baseline_result["test_accuracy"] * 100.0) if baseline_result["test_accuracy"] > 0 else 0.0

    artifacts = save_run_summary(run_dir, best_config, search_result, baseline_result, final_result, run_config)
    logger.info("run_complete=%s artifacts=%s", run_index + 1, artifacts)

    return {
        "run_dir": run_dir,
        "search_result": search_result,
        "baseline_result": baseline_result,
        "final_result": final_result,
        "best_config": best_config,
        "improvement": improvement,
        "relative_improvement": relative_improvement,
        "artifacts": artifacts,
        "config": config_to_dict(run_config),
    }


def copy_artifacts_to_root(run_artifacts, results_dir):
    for key in ["best_config_path", "summary_path", "accuracy_curve_path", "loss_curve_path", "global_best_path", "local_bests_path", "confusion_matrix_path"]:
        source = run_artifacts[key]
        target_name = os.path.basename(source)
        target = os.path.join(results_dir, target_name)
        shutil.copy2(source, target)

    best_model_source = os.path.join(os.path.dirname(run_artifacts["best_config_path"]), "best_model.pth")
    if os.path.exists(best_model_source):
        shutil.copy2(best_model_source, os.path.join(results_dir, "best_model.pth"))


def aggregate_results(runs_results):
    baseline_test = np.array([run["baseline_result"]["test_accuracy"] for run in runs_results], dtype=float)
    gwo_test = np.array([run["final_result"]["test_accuracy"] for run in runs_results], dtype=float)
    optimization_time = np.array([run["search_result"]["optimization_time"] for run in runs_results], dtype=float)
    training_time = np.array([run["final_result"]["training_time"] for run in runs_results], dtype=float)

    return {
        "baseline_test_mean": float(np.mean(baseline_test)),
        "baseline_test_std": float(np.std(baseline_test, ddof=0)),
        "gwo_test_mean": float(np.mean(gwo_test)),
        "gwo_test_std": float(np.std(gwo_test, ddof=0)),
        "optimization_time_mean": float(np.mean(optimization_time)),
        "optimization_time_std": float(np.std(optimization_time, ddof=0)),
        "training_time_mean": float(np.mean(training_time)),
        "training_time_std": float(np.std(training_time, ddof=0)),
    }


def write_root_summary(base_config, runs_results, aggregate):
    root_summary_path = os.path.join(base_config.results_dir, "final_results.txt")
    with open(root_summary_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"population size: {base_config.population_size}\n")
        file_handle.write(f"iteration count: {base_config.iteration_count}\n")
        file_handle.write(f"runs: {base_config.runs}\n")
        file_handle.write(f"mean baseline test accuracy: {aggregate['baseline_test_mean']:.6f}\n")
        file_handle.write(f"std baseline test accuracy: {aggregate['baseline_test_std']:.6f}\n")
        file_handle.write(f"mean GWO test accuracy: {aggregate['gwo_test_mean']:.6f}\n")
        file_handle.write(f"std GWO test accuracy: {aggregate['gwo_test_std']:.6f}\n")
        file_handle.write(f"mean optimization time (seconds): {aggregate['optimization_time_mean']:.2f}\n")
        file_handle.write(f"std optimization time (seconds): {aggregate['optimization_time_std']:.2f}\n")
        file_handle.write(f"mean training time (seconds): {aggregate['training_time_mean']:.2f}\n")
        file_handle.write(f"std training time (seconds): {aggregate['training_time_std']:.2f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--iteration-count", type=int, default=None)
    parser.add_argument("--search-epochs", type=int, default=None)
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    override_map = {
        "runs": args.runs,
        "population_size": args.population_size,
        "iteration_count": args.iteration_count,
        "search_epochs": args.search_epochs,
        "final_epochs": args.final_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "subset_size": args.subset_size,
        "random_seed": args.random_seed,
        "patience": args.patience,
        "dataset": args.dataset,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "num_workers": args.num_workers,
        "results_dir": args.results_dir,
        "data_dir": args.data_dir,
    }
    for key, value in override_map.items():
        if value is not None:
            setattr(config, key, value)

    ensure_dir(config.results_dir)
    logger = setup_logging(os.path.join(config.results_dir, "run.log"))
    write_json(os.path.join(config.results_dir, "config_used.json"), config_to_dict(config))
    write_search_space_file(config, os.path.join(config.results_dir, "search_space.json"))

    logger.info("experiment_start config=%s", config_to_dict(config))

    experiment_start = time.perf_counter()
    runs_results = []
    for run_index in range(config.runs):
        logger.info("starting_run=%s/%s", run_index + 1, config.runs)
        run_result = execute_single_run(config, run_index, logger)
        runs_results.append(run_result)

    total_experiment_time = time.perf_counter() - experiment_start
    root_artifacts = build_root_artifacts(config, runs_results, total_experiment_time)
    logger.info("root_artifacts=%s", root_artifacts)

    logger.info("experiment_complete")


if __name__ == "__main__":
    main()
