# app.py
import os
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import chainlit as cl

# --- LangChain / RAG bits ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =============================================================================
# Settings
# =============================================================================
try:
    from settings import settings as _ext  # type: ignore
except Exception:
    _ext = None


@dataclass
class Settings:
    docs_dir: Path = Path("./data")
    chroma_dir: Path = Path("./.chroma")
    chroma_collection: str = "vipo_bank_policies"

    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    max_k: int = 12
    rerank_top_n: int = 4
    min_relevance: float = 0.35


def _resolve_settings(ext=_ext) -> Settings:
    d = Settings()
    return Settings(
        docs_dir=Path(getattr(ext, "docs_dir", d.docs_dir)),
        chroma_dir=Path(getattr(ext, "chroma_dir", d.chroma_dir)),
        chroma_collection=getattr(ext, "chroma_collection", d.chroma_collection),
        openai_model=getattr(ext, "openai_model", d.openai_model),
        max_k=getattr(ext, "max_k", d.max_k),
        rerank_top_n=getattr(ext, "rerank_top_n", d.rerank_top_n),
        min_relevance=getattr(ext, "min_relevance", d.min_relevance),
    )


settings = _resolve_settings()
settings.chroma_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Embeddings / Vector store / Retriever
# =============================================================================
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectorstore = Chroma(
    collection_name=settings.chroma_collection,
    embedding_function=embeddings,
    persist_directory=str(settings.chroma_dir),
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.max_k})

CROSS_ENCODER_NAME = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
cross_encoder = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_NAME)
reranker = CrossEncoderReranker(model=cross_encoder, top_n=settings.rerank_top_n)

compressor_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=reranker,
)

# =============================================================================
# Helpers
# =============================================================================
def _similarity_gate(query: str, k: int = 4) -> Tuple[bool, List[str]]:
    """Quick relevance pre-check."""
    source_ids: List[str] = []
    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        if not results:
            return False, source_ids
        best_score = results[0][1]
        for doc, _ in results:
            sid = doc.metadata.get("source") or doc.metadata.get("path") or doc.metadata.get("file_name")
            if sid:
                source_ids.append(str(sid))
        return (best_score >= settings.min_relevance), source_ids
    except Exception:
        try:
            results = vectorstore.similarity_search_with_score(query, k=k)
            if not results:
                return False, source_ids
            best_dist = float(results[0][1])
            for doc, _ in results:
                sid = doc.metadata.get("source") or doc.metadata.get("path") or doc.metadata.get("file_name")
                if sid:
                    source_ids.append(str(sid))
            proxy_relevance = 1.0 / (1.0 + best_dist)
            return (proxy_relevance >= settings.min_relevance), source_ids
        except Exception:
            return False, source_ids


def format_docs(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source") or d.metadata.get("path") or d.metadata.get("file_name") or f"doc_{i}"
        chunk = (d.page_content or "").strip()
        if chunk:
            blocks.append(f"[{i}] Source: {src}\n{chunk}")
    return "\n\n".join(blocks)


# =============================================================================
# Prompt / LLM / Chain
# =============================================================================
SYSTEM_PROMPT = """
<|START_OF_TEXT|>
You are a concise, helpful assistant for banking and policy Q&A.
- Use ONLY the provided CONTEXT to answer. 
- If the answer is not in the CONTEXT, politely say: 
  "I don’t have that information in the provided context. Please refer to official sources or documentation for the most accurate details."
- Keep responses brief, accurate, and professional.
- When applicable, include sources in the format [1], [2].
<|END_OF_TEXT|>
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nCONTEXT:\n{context}\n\nAnswer:"),
    ]
)

llm = ChatOpenAI(model=settings.openai_model, temperature=0.2)
rag_chain = PROMPT | llm | StrOutputParser()

# =============================================================================
# Chainlit app
# =============================================================================
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    pass


@cl.on_chat_resume
async def on_chat_resume():
    history = cl.user_session.get("history", [])
    if not history:
        await cl.Message(content="Resuming chat. No prior history found.").send()
    else:
        recap = "\n".join(
            [f"**{'User' if turn['role']=='user' else 'Assistant'}:** {turn['content']}"
             for turn in history[-6:]]  # show last 3 turns
        )
        await cl.Message(content=f"Resuming chat. Here's your recent conversation:\n\n{recap}").send()


@cl.on_message
async def on_message(message: cl.Message):
    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Please enter a question.").send()
        return

    history = cl.user_session.get("history", [])

    # Gate check
    relevant_enough, _ = _similarity_gate(query, k=4)
    if not relevant_enough:
        await cl.Message(content="I don't know").send()
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": "I don't know"})
        cl.user_session.set("history", history)
        return

    # Retrieve docs
    try:
        docs = await cl.make_async(compressor_retriever.get_relevant_documents)(query)
    except Exception as e:
        err_msg = f"Retrieval error: {e}\nCheck your Chroma index at {settings.chroma_dir}."
        await cl.Message(content=err_msg).send()
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": err_msg})
        cl.user_session.set("history", history)
        return

    if not docs:
        await cl.Message(content="I don't know").send()
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": "I don't know"})
        cl.user_session.set("history", history)
        return

    context_text = format_docs(docs)

    # Build full prompt with history
    def build_prompt_with_history(history, question, context):
        conversation_str = ""
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            conversation_str += f"{role}: {turn['content']}\n"
        return f"{conversation_str}\nUser: {question}\n\nCONTEXT:\n{context}\n\nAnswer:"

    full_prompt = build_prompt_with_history(history, query, context_text)

    # Run LLM once to get full response
    try:
        response = await cl.make_async(rag_chain.invoke)(
            {"question": full_prompt, "context": ""}
        )
    except Exception as e:
        err_msg = f"Generation error: {e}"
        await cl.Message(content=err_msg).send()
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": err_msg})
        cl.user_session.set("history", history)
        return

    # Stream one word at a time
    msg = cl.Message(content="")
    await msg.send()

    for word in response.split():
        await msg.stream_token(word + " ")
        await asyncio.sleep(0.15)  # typing delay

    await msg.update()

    # Save history
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response})
    MAX_TURNS = 10
    if len(history) > MAX_TURNS * 2:
        history = history[-MAX_TURNS*2:]
    cl.user_session.set("history", history)


# # app.py
# import os
# from pathlib import Path
# from dataclasses import dataclass
# from typing import List, Tuple

# import chainlit as cl

# # --- LangChain / RAG bits ---
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser


# # =============================================================================
# # Settings (robust: merges external settings if available with sane defaults)
# # =============================================================================
# try:
#     # If you have settings.py exposing `settings`, we’ll merge from it.
#     from settings import settings as _ext  # type: ignore
# except Exception:
#     _ext = None


# @dataclass
# class Settings:
#     # Local data & vector store
#     docs_dir: Path = Path("./data")
#     chroma_dir: Path = Path("./.chroma")
#     chroma_collection: str = "vipo_bank_policies"

#     # OpenAI
#     openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

#     # Retrieval knobs
#     max_k: int = 12            # initial vector fetch
#     rerank_top_n: int = 4      # cross-encoder keeps top N
#     min_relevance: float = 0.35  # gate for "I don't know" (0..1)


# def _resolve_settings(ext=_ext) -> Settings:
#     d = Settings()
#     return Settings(
#         docs_dir=Path(getattr(ext, "docs_dir", d.docs_dir)),
#         chroma_dir=Path(getattr(ext, "chroma_dir", d.chroma_dir)),
#         chroma_collection=getattr(ext, "chroma_collection", d.chroma_collection),
#         openai_model=getattr(ext, "openai_model", d.openai_model),
#         max_k=getattr(ext, "max_k", d.max_k),
#         rerank_top_n=getattr(ext, "rerank_top_n", d.rerank_top_n),
#         min_relevance=getattr(ext, "min_relevance", d.min_relevance),
#     )


# settings = _resolve_settings()
# settings.chroma_dir.mkdir(parents=True, exist_ok=True)

# # Optional: quick config echo
# # print(
# #     f"[cfg] model={settings.openai_model}  k={settings.max_k}  "
# #     f"top_n={settings.rerank_top_n}  min_rel={settings.min_relevance}"
# # )
# # print(
# #     f"[cfg] chroma_dir={settings.chroma_dir}  collection={settings.chroma_collection}"
# # )

# # =============================================================================
# # Embeddings / Vector store / Retriever
# # =============================================================================
# EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# vectorstore = Chroma(
#     collection_name=settings.chroma_collection,
#     embedding_function=embeddings,
#     persist_directory=str(settings.chroma_dir),
# )

# base_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.max_k})

# CROSS_ENCODER_NAME = os.getenv(
#     "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
# )
# cross_encoder = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_NAME)
# reranker = CrossEncoderReranker(model=cross_encoder, top_n=settings.rerank_top_n)

# compressor_retriever = ContextualCompressionRetriever(
#     base_retriever=base_retriever,
#     base_compressor=reranker,
# )

# # =============================================================================
# # Helpers
# # =============================================================================
# def _similarity_gate(query: str, k: int = 4) -> Tuple[bool, List[str]]:
#     """
#     Quick pre-check using relevance scores if available, else distance heuristic.
#     Returns (is_relevant_enough, sample_source_ids).
#     """
#     source_ids: List[str] = []
#     try:
#         results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
#         if not results:
#             return False, source_ids
#         best_score = results[0][1]  # 0..1 (higher=better)
#         for doc, _ in results:
#             sid = (
#                 doc.metadata.get("source")
#                 or doc.metadata.get("path")
#                 or doc.metadata.get("file_name")
#             )
#             if sid:
#                 source_ids.append(str(sid))
#         return (best_score >= settings.min_relevance), source_ids
#     except Exception:
#         # Fallback to distance-based score if relevance API not supported
#         try:
#             results = vectorstore.similarity_search_with_score(query, k=k)
#             if not results:
#                 return False, source_ids
#             best_dist = float(results[0][1])
#             for doc, _ in results:
#                 sid = (
#                     doc.metadata.get("source")
#                     or doc.metadata.get("path")
#                     or doc.metadata.get("file_name")
#                 )
#                 if sid:
#                     source_ids.append(str(sid))
#             proxy_relevance = 1.0 / (1.0 + best_dist)  # map distance→[0,1]
#             return (proxy_relevance >= settings.min_relevance), source_ids
#         except Exception:
#             # If we can’t gauge relevance, be conservative
#             return False, source_ids


# def format_docs(docs) -> str:
#     """Join docs with lightweight source markers for the prompt."""
#     blocks = []
#     for i, d in enumerate(docs, start=1):
#         src = (
#             d.metadata.get("source")
#             or d.metadata.get("path")
#             or d.metadata.get("file_name")
#             or f"doc_{i}"
#         )
#         chunk = (d.page_content or "").strip()
#         if chunk:
#             blocks.append(f"[{i}] Source: {src}\n{chunk}")
#     return "\n\n".join(blocks)


# # =============================================================================
# # Prompt / LLM / Chain
# # =============================================================================
# SYSTEM_PROMPT = """
# <|START_OF_TEXT|>You are a concise, helpful assistant for bank/policy Q&A.
# Use ONLY the provided CONTEXT to answer. If the answer is not in the context, say: "I'm not sure about that kindly ask me anything related to bank policies and upload.
# Be brief, correct, and include a short list of sources as [1], [2] if applicable.
# <|END_OF_TEXT|>
# """

# PROMPT = ChatPromptTemplate.from_messages(
#     [
#         ("system", SYSTEM_PROMPT),
#         ("human", "Question: {question}\n\nCONTEXT:\n{context}\n\nAnswer:"),
#     ]
# )

# llm = ChatOpenAI(model=settings.openai_model, temperature=0.2)
# rag_chain = PROMPT | llm | StrOutputParser()

# # =============================================================================
# # Chainlit app
# # =============================================================================
# @cl.on_chat_start
# async def on_chat_start():
#     # Just show the logo without any welcome message
#     cl.user_session.set("history", [])
#     pass

# @cl.on_chat_resume
# async def on_chat_resume():
#     history = cl.user_session.get("history", [])
#     if not history:
#         await cl.Message(content="Resuming chat. No prior history found.").send()
#     else:
#         recap = "\n".join(
#             [f"**{'User' if turn['role']=='user' else 'Assistant'}:** {turn['content']}"
#              for turn in history[-6:]]  # show last 3 turns
#         )
#         await cl.Message(content=f"Resuming chat. Here's your recent conversation:\n\n{recap}").send()

# @cl.on_message
# async def on_message(message: cl.Message):
#     query = (message.content or "").strip()
#     if not query:
#         await cl.Message(content="Please enter a question.").send()
#         return

#     # Get conversation history from session (init if not present)
#     history = cl.user_session.get("history", [])
    
#     # Fast gate: avoid hallucinations if no relevant neighbors
#     relevant_enough, _ = _similarity_gate(query, k=4)
#     if not relevant_enough:
#         await cl.Message(content="I don't know").send()
#         history.append({"role": "user", "content": query})
#         history.append({"role": "assistant", "content": "I don't know"})
#         cl.user_session.set("history", history)
#         return

#     # Retrieve + rerank compressed context
#     try:
#         docs = await cl.make_async(compressor_retriever.get_relevant_documents)(query)
#     except Exception as e:
#         err_msg = f"Retrieval error: {e}\nCheck your Chroma index at {settings.chroma_dir}."
#         await cl.Message(content=err_msg).send()
#         history.append({"role": "user", "content": query})
#         history.append({"role": "assistant", "content": err_msg})
#         cl.user_session.set("history", history)
#         return

#     if not docs:
#         await cl.Message(content="I don't know").send()
#         history.append({"role": "user", "content": query})
#         history.append({"role": "assistant", "content": "I don't know"})
#         cl.user_session.set("history", history)
#         return

#     context_text = format_docs(docs)

#     # Build full prompt with history
#     def build_prompt_with_history(history, question, context):
#         conversation_str = ""
#         for turn in history:
#             role = "User" if turn["role"] == "user" else "Assistant"
#             conversation_str += f"{role}: {turn['content']}\n"
#         return f"{conversation_str}\nUser: {question}\n\nCONTEXT:\n{context}\n\nAnswer:"

#     full_prompt = build_prompt_with_history(history, query, context_text)

#     # Stream tokens to the UI
#     msg = cl.Message(content="")
#     await msg.send()
#     try:
#         async for token in rag_chain.astream({"question": full_prompt, "context": ""}):
#             if token:
#                 await msg.stream_token(token)
#         await msg.update()

#         # Save to history
#         history.append({"role": "user", "content": query})
#         history.append({"role": "assistant", "content": msg.content})
#         # Keep last N turns
#         MAX_TURNS = 10
#         if len(history) > MAX_TURNS * 2:
#             history = history[-MAX_TURNS*2:]
#         cl.user_session.set("history", history)
        
#     except Exception as e:
#         err_msg = f"Generation error: {e}"
#         await msg.update(content=err_msg)
#         history.append({"role": "user", "content": query})
#         history.append({"role": "assistant", "content": err_msg})
#         cl.user_session.set("history", history)




# # Settings class definition
# class Settings:
#     # Local data & vector store
#     docs_dir: Path = Path("./data")
#     chroma_dir: Path = Path("./.chroma")
#     chroma_collection: str = "vipo_bank_policies"

#     # OpenAI
#     openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
#     openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

#     # Optional: Google fallback
#     google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
#     google_llm_model: str = os.getenv("GOOGLE_LLM_MODEL", "gemini-1.5-pro")

# settings = Settings()

# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# def build_vectordb():
#     embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     return Chroma(
#         collection_name=settings.chroma_collection,
#         persist_directory=str(settings.chroma_dir),
#         embedding_function=embed,
#     )

# def build_retriever(vectordb, k=8):
#     """Build retriever with fallback to basic search if reranking fails"""
#     try:
#         # Try to use cross-encoder re-ranking
#         base_retriever = vectordb.as_retriever(search_kwargs={"k": max(k * 2, 12)})
        
#         try:
#             cross_encoder = HuggingFaceCrossEncoder(
#                 model_name="BAAI/bge-reranker-base",
#                 model_kwargs={"device": "cpu"},
#             )
#         except Exception:
#             try:
#                 cross_encoder = HuggingFaceCrossEncoder(
#                     model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
#                     model_kwargs={"device": "cpu"},
#                 )
#             except Exception:
#                 # If both rerankers fail, use basic retrieval
#                 print("⚠️  Reranking failed, using basic retrieval")
#                 return vectordb.as_retriever(search_kwargs={"k": k})

#         compressor = CrossEncoderReranker(model=cross_encoder, top_n=k)
#         return ContextualCompressionRetriever(
#             base_retriever=base_retriever,
#             base_compressor=compressor,
#         )
        
#     except Exception as e:
#         print(f"⚠️  Error building retriever: {e}, using basic retrieval")
#         return vectordb.as_retriever(search_kwargs={"k": k})

# def build_llm():
#     if settings.openai_api_key:
#         return ChatOpenAI(
#             model=settings.openai_model,
#             temperature=0.2,
#             api_key=settings.openai_api_key,
#         )
#     try:
#         from langchain_google_genai import ChatGoogleGenerativeAI
#         if settings.google_api_key:
#             return ChatGoogleGenerativeAI(
#                 model=settings.google_llm_model,
#                 temperature=0.2,
#                 google_api_key=settings.google_api_key,
#             )
#     except ImportError:
#         pass
#     raise RuntimeError("No LLM configured. Set OPENAI_API_KEY in .env")

# def format_context(docs):
#     blocks = []
#     for d in docs:
#         src = d.metadata.get("doc_name") or d.metadata.get("source") or "unknown"
#         page = d.metadata.get("page", d.metadata.get("page_number"))
#         tag = f"[source:{src}{f'#p{page}' if page is not None else ''}]"
#         blocks.append(f"{tag}\n{d.page_content}")
#     return "\n\n---\n\n".join(blocks)

# SYSTEM_PROMPT = """You are Vipo, a helpful assistant for banking policies and regulations.
# You must answer using ONLY the retrieved context from Bank of America documents.
# If the context doesn't contain the answer, say "I don't know".
# Keep answers concise and include inline [source:filename] citations next to the claims they support.
# """

# PROMPT = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT),
#     ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer with inline [source:filename] citations:"),
# ])

# @cl.on_chat_start
# async def start():
#     try:
#         vectordb = build_vectordb()
#         try:
#             count = vectordb._collection.count()  # may change across versions
#             if count == 0:
#                 await cl.Message(
#                     content=(
#                         "Your vector store is empty.\n\n"
#                         "Place PDFs/TXT/MD files in ./data and run: python ingest.py"
#                     )
#                 ).send()
#         except Exception:
#             pass

#         retriever = build_retriever(vectordb, k=8)
#         llm = build_llm()

#         cl.user_session.set("retriever", retriever)
#         cl.user_session.set("llm", llm)

#         await cl.Message(
#             content="Vipo Banking Assistant is ready. Ask about policies, fees, or regulations. Answers will include citations."
#         ).send()
#     except Exception as e:
#         await cl.Message(
#             content=(
#                 f"Error initializing: {e}\n"
#                 "Check your .env (OPENAI_API_KEY) and run python ingest.py."
#             )
#         ).send()

# @cl.on_message
# async def on_message(msg: cl.Message):
#     try:
#         retriever = cl.user_session.get("retriever")
#         llm = cl.user_session.get("llm")
#         if not retriever or not llm:
#             await cl.Message(content="Session not initialized. Please refresh.").send()
#             return

#         await cl.Message(content="Searching your documents...").send()

#         docs = await cl.make_async(retriever.invoke)(msg.content)
#         if not docs:
#             await cl.Message(content="No relevant documents found.").send()
#             return

#         context = format_context(docs)
#         chain = PROMPT | llm
#         resp = await cl.make_async(chain.invoke)({"question": msg.content, "context": context})

#         await cl.Message(content=resp.content).send()

#         elements = []
#         for d in docs:
#             src = d.metadata.get("doc_name") or d.metadata.get("source") or "unknown"
#             path = d.metadata.get("path")
#             page = d.metadata.get("page", d.metadata.get("page_number"))
#             if path:
#                 elements.append(cl.Text(name=f"{src}#p{page}", content=f"{path} : page {page}"))

#         if elements:
#             await cl.Message(content="Sources", elements=elements, author="Vipo").send()

#     except Exception as e:
#         await cl.Message(content=f"Error: {e}").send()
