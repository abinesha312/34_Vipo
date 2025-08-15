import json, pandas as pd
from pathlib import Path
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from rag import build_retriever, build_llm


TESTSET = Path(__file__).with_name("testset.jsonl")


def load_questions():
    return [json.loads(l) for l in TESTSET.read_text().splitlines() if l.strip()]


def run_eval():
    retriever = build_retriever(k=12)
    llm = build_llm()

    rows = []
    for ex in load_questions():
        q = ex["question"]
        docs = retriever.invoke(q)
        answer = llm.invoke(f"Answer briefly using this context only.\n\n{docs[0].page_content}\n\nQ: {q}").content
        rows.append({
            "question": q,
            "answer": answer,
            "contexts": [d.page_content for d in docs],
            "ground_truth": ex.get("reference", ""),
        })

    df = pd.DataFrame(rows)
    result = evaluate(
        df,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm if isinstance(llm, ChatGoogleGenerativeAI) else ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
    )
    print(result)


if __name__ == "__main__":
    run_eval()

