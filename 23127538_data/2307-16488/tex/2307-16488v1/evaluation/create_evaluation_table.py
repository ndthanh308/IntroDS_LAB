#!/usr/bin/env python3
import os
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
import yaml


CATEGORIES = ["Category A", "Category B", "Category C"]
METHODS = ["dexnet", "zeng", "ours"]
RESULT_NAMES = {
    "dexnet": "top_grasps.exr",
    "zeng": "zeng_predicted_target_large.png",
    "ours": "eval_result.png",
}
# specify the path to the root folder containing
# - one folder for each entry in CATEGORIES
# - one folder for each entry in METHODS
#   and each method folder containing a results folder for each category
RESULT_PATH = os.path.expanduser("TODO")


@dataclass
class SampleResult:
    object_class: str
    sample_id: str
    target_quality: np.ndarray
    predictions: Dict[str, np.ndarray]  # one entry per method in METHODS


def iterate_samples(category: str):
    samples_path = os.path.join(RESULT_PATH, category)
    for object_class in os.listdir(samples_path):
        object_class_path = os.path.join(samples_path, object_class)
        if not os.path.isdir(object_class_path):
            continue

        # iterate over individual samples
        for sample_id in os.listdir(object_class_path):
            sample_id_path = os.path.join(object_class_path, sample_id)
            if not os.path.isdir(sample_id_path):
                continue
            print(f"Sample {object_class} / {sample_id} ...")
            target_quality = cv2.imread(os.path.join(sample_id_path, "target_quality.png"))[..., 0] / 255
            assert target_quality.max() <= 1

            # iterate over methods
            results = {}
            for method in METHODS:
                result_path = os.path.join(RESULT_PATH, method, category, object_class, sample_id, RESULT_NAMES[method])
                _, ext = os.path.splitext(result_path)
                if ext == ".png":
                    result = cv2.imread(result_path)> 0
                elif ext == ".exr":
                    result = cv2.imread(result_path, cv2.IMREAD_UNCHANGED) > 0
                if len(result.shape) == 3:
                    result = result[..., 0]
                assert result.sum() <= 20, f"{method} has {result.sum()} results"
                results[method] = result

            yield SampleResult(object_class, sample_id, target_quality, results)


def evaluate():
    result_summary = {}
    for category in CATEGORIES:
        result_details = {}
        for sample in iterate_samples(category):
            sample_result = {}
            for method in METHODS:
                result = sample.predictions[method]
                quality_vals = sample.target_quality[result]
                quality_sum = quality_vals.sum()
                quality_max = quality_vals.max()
                feasible_sum = (quality_vals > 0).sum()
                num_grasps = result.sum()

                sample_result[method] = {
                    "num_grasps": num_grasps,
                    "success_rate": feasible_sum / num_grasps if num_grasps > 0 else 0,
                    "quality_avg": quality_sum / feasible_sum if feasible_sum > 0 else 0,
                    "quality_max": quality_max,
                }

                print(f"  {method}")
                print("    Num grasps:", num_grasps)
                print("    Success rate:", feasible_sum / num_grasps if num_grasps > 0 else 0)
                print("    Avg quality:", quality_sum / feasible_sum if feasible_sum > 0 else 0)
                print("    Max quality:", quality_max)

            result_details[f"{sample.object_class}/{sample.sample_id}"] = sample_result

        with open(os.path.join(RESULT_PATH, f"{category}.yaml"), "w") as f:
            yaml.dump(result_details, f)

        result_summary[category] = {}
        for method in METHODS:
            result_summary[category][method] = {
                "success_rate": sum(
                    result[method]["success_rate"] for result in result_details.values()
                ) / len(result_details) * 100,
                "quality_avg": sum(
                    result[method]["quality_avg"] for result in result_details.values()
                ) / len(result_details) * 100,
            }
    print(result_summary)

    tex = "\\begin{tabular}{lccccccc}\n\\toprule\n"
    tex += "&" + "".join(f" {c} & &" for c in CATEGORIES) + "\\\\\n"
    tex += "&" + "".join(" AGQ [\\%] & SR [\\%] &" for _ in CATEGORIES) + "\\\\\n"
    tex += "\\midrule\n"
    for method in METHODS:
        tex += method + " &"
        for category in CATEGORIES:
            res = result_summary[category][method]
            tex += f" {res['quality_avg']:.1f} & {res['success_rate']:.1f} &"
        tex += " \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    with open(os.path.join(RESULT_PATH, "evaluation_table.tex"), "w") as f:
        f.write(tex)


if __name__ == "__main__":
    assert os.path.exists(RESULT_PATH), f"Result path {RESULT_PATH} does not exist"
    evaluate()
