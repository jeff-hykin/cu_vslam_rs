#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""Compute cuVSLAM evaluation KPIs from cuvslam_app stats and render a Markdown table.

Runner-local port of the OSMO osmo_reporter (full_kpis_report.py). The KPI math
(ATE / ARE / Kabsch / Losts / FPS, per dataset, ODOM + SLAM) is unchanged; the
OSMO-specific output paths and the Slack webhook notification are removed so the
script runs on the GitHub Actions gpu runner with only the standard library.

Input: a stats folder containing one subfolder per dataset run, each holding a
timestamped folder with stats/all_stats.json (the layout cuvslam_app writes).
Output: a KPI JSON file plus a pipe-delimited <json>.table that renders as native
Markdown for the nightly GitHub Release body.
"""

from argparse import ArgumentParser
import json
import os
import glob

NO_DIFF_METRICS = {"FPS"}

DATASET_DISPLAY_ALIASES = {"TARTAN_FLAKY": "TARTAN_F"}


def display_dataset_key(key):
    """TARTAN_FLAKY-STEREO_ODOM -> TARTAN_F-STEREO_ODOM (display only)."""
    for full, short in DATASET_DISPLAY_ALIASES.items():
        if key.startswith(full + "-"):
            return short + key[len(full):]
    return key


def get_unit(metric, metric_units={"ATE": "%", "ARE": "º/m", "Kabsch": "", "TrackingLosts": "", "FPS": "Hz"}):
    for unit_key, unit_value in metric_units.items():
        if unit_key in metric:
            return unit_value
    return ''


def get_display_name(metric):
    """Get display name for metric (e.g., 'TrackingLosts' -> 'Losts')."""
    display_names = {
        "TrackingLosts": "Losts",
        "diff TrackingLosts": "diff Losts"
    }
    return display_names.get(metric, metric)


def parse_all_stats_json(json_path):
    """Parse the all_stats.json file.

    Returns:
        list: List of stat dictionaries
    """
    try:
        with open(json_path, 'r') as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        print(f'Warning: failed to parse JSON file {json_path}: {e}')
        return None


def odometry_mode_to_type(odometry_mode):
    """Convert odometry_mode string to dataset type.

    Args:
        odometry_mode: String like "OdometryMode.Multicamera". Matching is
            case-insensitive, so command-line values like "multicamera" map the
            same way. None or non-string values fall back to STEREO.

    Returns:
        str: Dataset type (MONO, STEREO, VIO, RGBD)
    """
    normalized = str(odometry_mode).lower()
    if 'multicamera' in normalized:
        return 'STEREO'
    elif 'mono' in normalized:
        return 'MONO'
    elif 'inertial' in normalized:
        return 'VIO'
    elif 'rgbd' in normalized:
        return 'RGBD'
    else:
        return 'STEREO'


def load_baseline_ranges(path):
    """Load the committed baseline-ranges json.

    Returns the dict of per-KPI range specs (the "kpis" block, or the top-level
    dict if no "kpis" key), or an empty dict on any error. Never raises.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load baseline ranges {path}: {e}")
        return {}
    if not isinstance(data, dict):
        print(f"Warning: baseline ranges {path} is not a json object; skipping drift check")
        return {}
    ranges = data.get("kpis", data)
    return ranges if isinstance(ranges, dict) else {}


def safe_float(value):
    """Best-effort float conversion. Returns None for None, NaN, or non-numeric
    values (e.g. a malformed string in the committed ranges file)."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # drop NaN


def evaluate_drift(kpis_dict, ranges):
    """Compare each computed KPI against its expected value +/- tolerance.

    Soft check only: returns a list of (key, status, detail) rows and never
    raises, even on malformed baseline entries. Statuses: WITHIN (in range),
    DRIFT (out of range), SKIPPED (uncalibrated/malformed), MISSING (no value
    this run).
    """
    rows = []
    for key in sorted(ranges):
        try:
            spec = ranges[key] if isinstance(ranges[key], dict) else {}
            if key not in kpis_dict:
                rows.append((key, "MISSING", "no value produced this run"))
                continue
            actual = safe_float(kpis_dict[key])
            if actual is None:
                rows.append((key, "SKIPPED", f"non-numeric actual value: {kpis_dict[key]!r}"))
                continue
            raw_expected = spec.get("expected")
            expected = safe_float(raw_expected)
            if expected is None:
                detail = (f"uncalibrated (actual={actual:.4g})" if raw_expected is None
                          else f"non-numeric expected={raw_expected!r} (actual={actual:.4g})")
                rows.append((key, "SKIPPED", detail))
                continue
            if spec.get("tol_abs") is not None:
                tol = safe_float(spec.get("tol_abs"))
            else:
                tol_pct = safe_float(spec.get("tol_pct"))
                tol = None if tol_pct is None else abs(expected) * tol_pct / 100.0
            tol = abs(tol) if tol is not None else 0.0
            low, high = expected - tol, expected + tol
            status = "WITHIN" if low <= actual <= high else "DRIFT"
            rows.append((key, status, f"actual={actual:.4g} expected={expected:.4g} range=[{low:.4g}, {high:.4g}]"))
        except Exception as exc:
            rows.append((key, "SKIPPED", f"error evaluating spec {ranges.get(key)!r}: {exc}"))
    return rows


def process_dataset_folder(dataset_folder_path):
    """Process a dataset folder to extract metrics for ODOM and SLAM.

    Args:
        dataset_folder_path: Path to dataset folder (e.g., kitti-vio_slam_gt)

    Returns:
        dict: Dictionary with dataset metrics
    """
    dataset_name = os.path.basename(dataset_folder_path).split('-')[0].upper()

    timestamped_folders = glob.glob(os.path.join(dataset_folder_path, '*'))
    timestamped_folders = [f for f in timestamped_folders if os.path.isdir(f)]

    if not timestamped_folders:
        print(f'Warning: no timestamped folders found in {dataset_folder_path}')
        return None

    latest_folder = max(timestamped_folders, key=os.path.getmtime)
    stats_folder = os.path.join(latest_folder, 'stats')

    if not os.path.exists(stats_folder):
        print(f'Warning: stats folder not found in {latest_folder}')
        return None

    out_dict = {}

    all_stats_json = os.path.join(stats_folder, 'all_stats.json')
    if not os.path.exists(all_stats_json):
        print(f'Warning: all_stats.json not found in {stats_folder}')
        return None

    all_stats = parse_all_stats_json(all_stats_json)
    if not all_stats:
        print(f'Warning: failed to parse all_stats.json in {stats_folder}')
        return None

    if all_stats and 'odometry_mode' in all_stats[0]:
        dataset_type = odometry_mode_to_type(all_stats[0]['odometry_mode'])
        print(f'  Detected dataset type: {dataset_type} (from odometry_mode: {all_stats[0]["odometry_mode"]})')
    else:
        print(f'  Warning: odometry_mode not found in JSON, falling back to folder name parsing')
        folder_name = os.path.basename(dataset_folder_path).lower()
        if 'mono' in folder_name:
            dataset_type = 'MONO'
        elif 'vio' in folder_name or 'imu' in folder_name:
            dataset_type = 'VIO'
        elif 'rgbd' in folder_name or 'depth' in folder_name:
            dataset_type = 'RGBD'
        else:
            dataset_type = 'STEREO'

    odom_stats = [s for s in all_stats if 'ODOM' in s.get('sequence_title', '').upper()]
    slam_stats = [s for s in all_stats if 'SLAM' in s.get('sequence_title', '').upper()]

    if odom_stats:
        avg_translation_error = sum(s.get('gt_av_translation_error', 0) for s in odom_stats) / len(odom_stats)
        avg_rotation_error = sum(s.get('gt_av_rotation_error', 0) for s in odom_stats) / len(odom_stats)
        avg_kabsch = sum(s.get('gt_simple_error', 0) for s in odom_stats) / len(odom_stats)
        avg_fps = sum(s.get('average_fps', 0) for s in odom_stats) / len(odom_stats)
        total_tracking_losts = sum(s.get('num_tracking_losts', 0) for s in odom_stats if s.get('num_tracking_losts', -1) >= 0)

        out_dict[f"{dataset_name}_ATE_{dataset_type}_ODOM"] = avg_translation_error
        out_dict[f"{dataset_name}_ARE_{dataset_type}_ODOM"] = avg_rotation_error
        out_dict[f"{dataset_name}_Kabsch_{dataset_type}_ODOM"] = avg_kabsch
        out_dict[f"{dataset_name}_FPS_{dataset_type}_ODOM"] = avg_fps
        out_dict[f"{dataset_name}_TrackingLosts_{dataset_type}_ODOM"] = total_tracking_losts

    if slam_stats:
        avg_translation_error = sum(s.get('gt_av_translation_error', 0) for s in slam_stats) / len(slam_stats)
        avg_rotation_error = sum(s.get('gt_av_rotation_error', 0) for s in slam_stats) / len(slam_stats)
        avg_kabsch = sum(s.get('gt_simple_error', 0) for s in slam_stats) / len(slam_stats)
        avg_fps = sum(s.get('average_fps', 0) for s in slam_stats) / len(slam_stats)
        total_tracking_losts = sum(s.get('num_tracking_losts', 0) for s in slam_stats if s.get('num_tracking_losts', -1) >= 0)

        out_dict[f"{dataset_name}_ATE_{dataset_type}_SLAM"] = avg_translation_error
        out_dict[f"{dataset_name}_ARE_{dataset_type}_SLAM"] = avg_rotation_error
        out_dict[f"{dataset_name}_Kabsch_{dataset_type}_SLAM"] = avg_kabsch
        out_dict[f"{dataset_name}_FPS_{dataset_type}_SLAM"] = avg_fps
        out_dict[f"{dataset_name}_TrackingLosts_{dataset_type}_SLAM"] = total_tracking_losts

    return out_dict


def organize_data(data, required_metrics, prev_data={}):
    """Organize data by dataset and metric.

    Args:
        data: Dictionary with keys like "KITTI_ATE_STEREO_ODOM"
        required_metrics: List of metrics to include (e.g., ["ATE", "ARE", "Kabsch"])
        prev_data: Previous data for comparison

    Returns:
        Dictionary organized by dataset with metrics
    """
    if not isinstance(prev_data, dict):
        print(f"Warning: prev_data is not a dictionary (type: {type(prev_data)}), using empty dict")
        prev_data = {}

    organized_data = {}

    for key, value in data.items():
        parts = key.split('_')

        if len(parts) < 4:
            continue

        odom_type = parts[-1]
        dataset_type = parts[-2]
        metric = parts[-3]
        dataset_name = '_'.join(parts[:-3])

        dataset_key = f"{dataset_name}-{dataset_type}_{odom_type}"

        if metric in required_metrics:
            if dataset_key not in organized_data:
                organized_data[dataset_key] = {}

            if metric == "TrackingLosts":
                organized_data[dataset_key][metric] = int(value)
            elif metric == "FPS":
                organized_data[dataset_key][metric] = round(value, 1)
            else:
                organized_data[dataset_key][metric] = round(value, 4)

            if metric in NO_DIFF_METRICS:
                pass
            elif metric == "TrackingLosts":
                if key in prev_data:
                    organized_data[dataset_key]["diff " + metric] = int(value - prev_data[key])
                else:
                    organized_data[dataset_key]["diff " + metric] = "NA"
            elif ("MONO" in dataset_key) and ("ATE" in metric):
                organized_data[dataset_key]["diff " + metric] = "NA"
            elif key in prev_data:
                organized_data[dataset_key]["diff " + metric] = round(value - prev_data[key], 4)
            else:
                organized_data[dataset_key]["diff " + metric] = "NA"

    for dataset_key, metrics_dict in organized_data.items():
        for metric in required_metrics:
            if metric not in metrics_dict:
                organized_data[dataset_key][metric] = "NA"
            if ("MONO" in dataset_key) and ("ATE" in metric):
                organized_data[dataset_key][metric] = "NA"

    return organized_data


def create_table(organized_data, required_metrics):
    diffable = [m for m in required_metrics if m not in NO_DIFF_METRICS]
    nondiff = [m for m in required_metrics if m in NO_DIFF_METRICS]
    columns = diffable + ["diff " + i for i in diffable] + nondiff

    display_keys = {d: display_dataset_key(d) for d in organized_data.keys()}
    ds_w = max([len("Dataset")] + [len(v) for v in display_keys.values()])

    header = f"| {'Dataset':<{ds_w}} |" + "|".join(f"{get_display_name(metric) + ',' + get_unit(metric):<12}" for metric in columns) + "|\n"
    separator = "|" + "-" * (ds_w + 2) + "|" + "------------|" * len(columns) + "\n"

    rows = []
    for dataset, metrics in organized_data.items():
        if all(metric in metrics for metric in columns):
            row = f"| {display_keys[dataset]:<{ds_w}} | " + " | ".join(f"{metrics[metric]:<10}" for metric in columns) + " |"
            rows.append(row)

    table = header + separator + "\n".join(rows)
    return table


def main():
    print("=============================\nKPI report generator is up!")
    parser = ArgumentParser()
    parser.add_argument('-s', '--stat_folder', type=str, help='full path to folder with stat_results folders (e.g., .../stat_results)')
    parser.add_argument('-j', '--out_kpi_json', type=str, help='full path to output KPI json')
    parser.add_argument('-d', '--run_id', type=str, help='run id label for logs/report', default="")
    parser.add_argument('-k', '--prev_kpi', type=str, help='path to previous KPI json file for diff calculation', default="")
    parser.add_argument('-b', '--baseline_ranges', type=str, help='path to committed KPI baseline-ranges json for a soft drift check', default="")
    args = parser.parse_args()

    kpis_dict = {}

    required_metrics = ["ATE", "ARE", "Kabsch", "TrackingLosts", "FPS"]

    if not os.path.exists(args.stat_folder):
        raise ValueError(f'Stat folder does not exist: {args.stat_folder}')

    dataset_folders = [
        os.path.join(args.stat_folder, d)
        for d in os.listdir(args.stat_folder)
        if os.path.isdir(os.path.join(args.stat_folder, d))
    ]

    if not dataset_folders:
        raise ValueError(f'No dataset folders found in: {args.stat_folder}')

    def sort_key(folder):
        name = os.path.basename(folder).lower()
        if 'mono' in name:
            return (0, name)
        elif 'stereo' in name and 'vio' not in name:
            return (1, name)
        elif 'vio' in name:
            return (2, name)
        elif 'rgbd' in name:
            return (3, name)
        else:
            return (4, name)

    dataset_folders = sorted(dataset_folders, key=sort_key)

    print(f"Processing {len(dataset_folders)} dataset folders...")
    for dataset_folder in dataset_folders:
        print(f"  Processing: {os.path.basename(dataset_folder)}")
        result = process_dataset_folder(dataset_folder)
        if result:
            kpis_dict.update(result)

    if not kpis_dict:
        raise ValueError('output KPI json is empty, please check the input stat files format')

    os.makedirs(os.path.dirname(args.out_kpi_json), exist_ok=True)

    with open(args.out_kpi_json, 'w') as outfile:
        json.dump(kpis_dict, outfile, indent=4)
        print("JSON report was successfully saved at", args.out_kpi_json)

    prev_kpi_dict = {}
    if os.path.isfile(args.prev_kpi):
        print("Load previous KPI data from", args.prev_kpi)
        try:
            with open(args.prev_kpi, 'r') as f:
                prev_kpi_dict = json.load(f)

            if isinstance(prev_kpi_dict, dict) and prev_kpi_dict:
                print(f"Loaded {len(prev_kpi_dict)} metrics from previous run")
            else:
                print("Warning: Previous KPI file is empty or invalid, ignoring previous data")
                prev_kpi_dict = {}
        except Exception as e:
            print(f"Warning: Failed to load previous KPI data: {e}")
            print("Continuing without previous data comparison")
            prev_kpi_dict = {}

    organized_data = organize_data(kpis_dict, required_metrics, prev_kpi_dict)
    table = create_table(organized_data, required_metrics)
    with open(args.out_kpi_json + '.table', "w") as file:
        file.write(table)

    print("=============================\nRun " + args.run_id + " finished with following results:\n")
    print(table)

    if args.baseline_ranges:
        ranges = load_baseline_ranges(args.baseline_ranges)
        if ranges:
            rows = evaluate_drift(kpis_dict, ranges)
            drift_lines = ["KPI drift check (soft, informational only) vs " + args.baseline_ranges]
            for key, status, detail in rows:
                drift_lines.append(f"  [{status:7}] {key}: {detail}")
            n_drift = sum(1 for _, status, _ in rows if status == "DRIFT")
            n_calibrated = sum(1 for _, status, _ in rows if status in ("WITHIN", "DRIFT"))
            if n_calibrated == 0:
                drift_lines.append("  (no calibrated KPIs yet; seed 'expected' values in the ranges file)")
            elif n_drift:
                drift_lines.append(f"  {n_drift} KPI(s) outside expected range (not failing the job).")
            drift_report = "\n".join(drift_lines)
            print("\n" + drift_report)
            with open(args.out_kpi_json + '.drift', 'w') as f:
                f.write(drift_report + "\n")
        else:
            print(f"Warning: no usable baseline ranges in {args.baseline_ranges}; skipping drift check.")


if __name__ == "__main__":
    main()
