from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from settings import settings


COLLECTION = "vipo_bank_policies"


def build_retriever(k: int = 8):
    # Must match the embedding model used during ingestion
    model_id = "manu/sentence_croissant_alpha_v0.2"
    embed_query = HuggingFaceEmbeddings(model_name=model_id)
    vectordb = Chroma(
        collection_name=COLLECTION,
        persist_directory=settings.chroma_dir,
        embedding_function=embed_query,
    )

    base = vectordb.as_retriever(search_kwargs={"k": max(k, settings.rerank_top_k)})

    cross_encoder = HuggingFaceCrossEncoder(model_name=settings.reranker_model, model_kwargs={"device": "cpu"})
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=settings.rerank_top_k)
    reranking = ContextualCompressionRetriever(base_retriever=base, base_compressor=compressor)

    return reranking


def build_llm():
    # Prefer Gemini if GOOGLE_API_KEY set, else fall back
    if settings.google_api_key:
        return ChatGoogleGenerativeAI(model=settings.google_llm_model, temperature=0.2)
    if settings.openai_api_key:
        return ChatOpenAI(model=settings.openai_model, temperature=0.2)
    # Last resort: try a local LLM via openai-compatible endpoint, left as exercise
    raise RuntimeError("No LLM configured. Set GOOGLE_API_KEY or OPENAI_API_KEY.")


SYSTEM = (
    "You are Vipo, a concise assistant for banking policies. "
    "Answer using ONLY the retrieved context. If unsure, say you don't know."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}\n\nGive a short, accurate answer with inline [source:filename] cites.",
    ),
])


def format_context(docs: List):
    lines = []
    for d in docs:
        # Prefer filename if present, else source type
        name = d.metadata.get("doc_name") or d.metadata.get("source") or "unknown"
        page = d.metadata.get("page", d.metadata.get("page_number"))
        tag = f"[source:{name}{f'#p{page}' if page is not None else ''}]"
        lines.append(f"{tag}\n{d.page_content}")
    return "\n\n---\n\n".join(lines)

