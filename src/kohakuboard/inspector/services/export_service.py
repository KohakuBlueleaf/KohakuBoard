"""Export service - export data to various formats"""

from pathlib import Path

import pandas as pd


class ExportService:
    """Service for exporting board data to various formats"""

    @staticmethod
    def export_metrics_to_parquet(
        data_service, metric_names: list[str], output_path: str | Path
    ):
        """Export metrics to Parquet file

        Args:
            data_service: DataService instance
            metric_names: List of metric names to export
            output_path: Output file path
        """
        # Combine all metrics into DataFrame
        combined_data = {}

        for metric in metric_names:
            data = data_service.get_metric_data(metric)
            combined_data[f"{metric}"] = data["values"]

        # Use first metric for index columns
        if metric_names:
            first_data = data_service.get_metric_data(metric_names[0])
            combined_data["step"] = first_data["steps"]
            combined_data["global_step"] = first_data["global_steps"]
            combined_data["timestamp"] = first_data["timestamps"]

        df = pd.DataFrame(combined_data)
        df.to_parquet(output_path, index=False)

    @staticmethod
    def export_metrics_to_csv(
        data_service, metric_names: list[str], output_path: str | Path
    ):
        """Export metrics to CSV file"""
        combined_data = {}

        for metric in metric_names:
            data = data_service.get_metric_data(metric)
            combined_data[f"{metric}"] = data["values"]

        if metric_names:
            first_data = data_service.get_metric_data(metric_names[0])
            combined_data["step"] = first_data["steps"]
            combined_data["global_step"] = first_data["global_steps"]
            combined_data["timestamp"] = first_data["timestamps"]

        df = pd.DataFrame(combined_data)
        df.to_csv(output_path, index=False)

    @staticmethod
    def export_metrics_to_json(
        data_service, metric_names: list[str], output_path: str | Path
    ):
        """Export metrics to JSON file"""
        import json

        combined_data = {}

        for metric in metric_names:
            data = data_service.get_metric_data(metric)
            combined_data[metric] = {
                "steps": data["steps"],
                "global_steps": data["global_steps"],
                "timestamps": data["timestamps"],
                "values": data["values"],
            }

        with open(output_path, "w") as f:
            json.dump(combined_data, f, indent=2)
