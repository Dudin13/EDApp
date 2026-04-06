# training_validator.py

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class EventDetectionTrainer:
    """Sistema de entrenamiento y validación para detección de eventos."""

    def __init__(self, ground_truth_dir: Path):
        self.ground_truth_dir = ground_truth_dir
        self.ground_truth_dir.mkdir(exist_ok=True)

    def create_synthetic_dataset(self, video_path: str, num_samples: int = 100) -> pd.DataFrame:
        """Crea dataset sintético para entrenamiento."""
        # Simular eventos para testing
        synthetic_events = []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        for i in range(num_samples):
            # Generar eventos sintéticos aleatorios
            timestamp = np.random.uniform(10, duration - 10)  # Evitar extremos
            event_type = np.random.choice(["Gol", "Corner", "Tiro a puerta", "Pase", "Falta"])

            synthetic_events.append({
                "video_id": Path(video_path).stem,
                "timestamp": timestamp,
                "event_type": event_type,
                "confidence": np.random.uniform(0.7, 1.0),
                "is_synthetic": True
            })

        cap.release()
        return pd.DataFrame(synthetic_events)

    def collect_ground_truth(self, video_path: str, events: List[Dict]) -> pd.DataFrame:
        """Recopila ground truth manual para validación."""
        gt_data = []

        for event in events:
            gt_data.append({
                "video_id": Path(video_path).stem,
                "timestamp": event["timestamp"],
                "event_type": event["action"],
                "confidence": event.get("confidence", 1.0),
                "validated": True,
                "manual_annotation": True
            })

        return pd.DataFrame(gt_data)

    def validate_predictions(self, predictions: pd.DataFrame,
                           ground_truth: pd.DataFrame, tolerance_seconds: float = 2.0):
        """Valida predicciones contra ground truth."""

        def find_matches(pred_row, gt_df):
            """Encuentra matches dentro de tolerancia temporal."""
            matches = gt_df[
                (abs(gt_df["timestamp"] - pred_row["timestamp"]) <= tolerance_seconds) &
                (gt_df["event_type"] == pred_row["event_type"])
            ]
            return len(matches) > 0

        # Calcular métricas
        predictions["is_correct"] = predictions.apply(
            lambda row: find_matches(row, ground_truth), axis=1
        )

        # Reporte de clasificación
        y_true = []
        y_pred = []

        for _, pred in predictions.iterrows():
            y_pred.append(pred["event_type"])
            # Buscar ground truth correspondiente
            gt_match = ground_truth[
                abs(ground_truth["timestamp"] - pred["timestamp"]) <= tolerance_seconds
            ]
            if len(gt_match) > 0:
                y_true.append(gt_match.iloc[0]["event_type"])
            else:
                y_true.append("No Event")  # False positive

        print("=== VALIDATION REPORT ===")
        print(f"Total predictions: {len(predictions)}")
        print(f"Correct predictions: {predictions['is_correct'].sum()}")
        print(".2f")

        if len(set(y_true)) > 1:
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, zero_division=0))

        return {
            "accuracy": predictions["is_correct"].mean(),
            "total_predictions": len(predictions),
            "correct_predictions": predictions["is_correct"].sum()
        }

    def generate_training_report(self, validation_results: Dict, output_path: Path):
        """Genera reporte visual de entrenamiento."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Gráfico de accuracy
        accuracies = [validation_results["accuracy"]]
        ax1.bar(["Current Model"], accuracies, color='skyblue')
        ax1.set_ylim(0, 1)
        ax1.set_title('Detection Accuracy')
        ax1.set_ylabel('Accuracy')

        # Gráfico de distribución de eventos
        event_counts = validation_results.get("event_distribution", {})
        if event_counts:
            ax2.pie(event_counts.values(), labels=event_counts.keys(), autopct='%1.1f%%')
            ax2.set_title('Event Distribution')

        plt.tight_layout()
        plt.savefig(output_path / "training_report.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Reporte textual
        report = f"""
        TRAINING VALIDATION REPORT
        ==========================

        Overall Accuracy: {validation_results['accuracy']:.2%}
        Total Predictions: {validation_results['total_predictions']}
        Correct Predictions: {validation_results['correct_predictions']}

        Generated: {output_path / "training_report.png"}
        """

        with open(output_path / "training_report.txt", "w") as f:
            f.write(report)