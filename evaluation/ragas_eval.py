import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
sys.path.append(".")

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from pipeline.rag_chain import ask
from evaluation.golden_dataset import GOLDEN_DATASET


def run_evaluation():
    print("\n🔍 Running RAGAS Evaluation...")
    print(f"Evaluating {len(GOLDEN_DATASET)} questions\n")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(GOLDEN_DATASET):
        print(f"Processing question {i+1}/{len(GOLDEN_DATASET)}: {item['question'][:50]}...")
        result = ask(item["question"], top_k=10)

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([s["content"] for s in result["sources"]])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Use Groq instead of OpenAI
    groq_llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",  # smaller, faster, uses fewer tokens
        temperature=0
    )

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    ragas_llm = LangchainLLMWrapper(groq_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        answer_correctness
    ]

    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    print("\n⚙️  Calculating RAGAS metrics...")
    scores = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        run_config={"max_workers": 1, "timeout": 120}
    )

    print("\n" + "=" * 50)
    print("📊 RAGAS EVALUATION RESULTS")
    print("=" * 50)
    print(f"Faithfulness:       {scores['faithfulness']:.3f}")
    print(f"Answer Relevancy:   {scores['answer_relevancy']:.3f}")
    print(f"Context Precision:  {scores['context_precision']:.3f}")
    print(f"Answer Correctness: {scores['answer_correctness']:.3f}")
    print("=" * 50)

    avg = (scores['faithfulness'] + scores['answer_relevancy'] +
           scores['context_precision'] + scores['answer_correctness']) / 4
    print(f"Overall Score:      {avg:.3f}")
    print("=" * 50)

    return scores


if __name__ == "__main__":
    run_evaluation()