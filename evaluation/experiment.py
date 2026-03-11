import os
import sys
sys.path.append(".")

import mlflow
import mlflow.sklearn
from datetime import datetime


def log_experiment(config: dict, scores: dict):
    mlflow.set_tracking_uri("./mlflow_runs")
    mlflow.set_experiment("FinRAG-Evaluation")

    run_name = f"{config['experiment_name']}_{datetime.now().strftime('%H%M%S')}"

    with mlflow.start_run(run_name=run_name):

        # Log configuration
        mlflow.log_params({
            "chunk_size": config.get("chunk_size", 512),
            "chunk_overlap": config.get("chunk_overlap", 50),
            "embedding_model": config.get("embedding_model", "all-MiniLM-L6-v2"),
            "llm_model": config.get("llm_model", "llama-3.1-8b-instant"),
            "top_k": config.get("top_k", 10),
            "num_questions": config.get("num_questions", 5),
            "document": config.get("document", "apple_10k.pdf")
        })

        # Log scores
        for metric, value in scores.items():
            if value is not None and str(value) != "nan":
                mlflow.log_metric(metric, float(value))

        print(f"✅ Experiment logged: {run_name}")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")


def log_manual_experiment(experiment_name: str, chunk_size: int,
                           embedding_model: str, top_k: int, scores: dict):
    config = {
        "experiment_name": experiment_name,
        "chunk_size": chunk_size,
        "embedding_model": embedding_model,
        "top_k": top_k
    }
    log_experiment(config, scores)


if __name__ == "__main__":
    # Log a sample experiment with dummy scores to test MLflow
    print("\n📊 Testing MLflow experiment tracking...\n")

    experiments = [
        {
            "config": {
                "experiment_name": "baseline",
                "chunk_size": 512,
                "embedding_model": "all-MiniLM-L6-v2",
                "llm_model": "llama-3.1-8b-instant",
                "top_k": 10
            },
            "scores": {
                "faithfulness": 0.82,
                "answer_relevancy": 0.75,
                "context_precision": 0.70,
                "answer_correctness": 0.66
            }
        },
        {
            "config": {
                "experiment_name": "larger_chunks",
                "chunk_size": 1024,
                "embedding_model": "all-MiniLM-L6-v2",
                "llm_model": "llama-3.1-8b-instant",
                "top_k": 10
            },
            "scores": {
                "faithfulness": 0.78,
                "answer_relevancy": 0.80,
                "context_precision": 0.65,
                "answer_correctness": 0.71
            }
        },
        {
            "config": {
                "experiment_name": "smaller_chunks",
                "chunk_size": 256,
                "embedding_model": "all-MiniLM-L6-v2",
                "llm_model": "llama-3.1-8b-instant",
                "top_k": 10
            },
            "scores": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.70,
                "context_precision": 0.75,
                "answer_correctness": 0.62
            }
        }
    ]

    for exp in experiments:
        log_experiment(exp["config"], exp["scores"])

    print("\n✅ All experiments logged!")
    print("Run 'mlflow ui' to view the dashboard\n")