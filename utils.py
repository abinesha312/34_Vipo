from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)


def chunk_documents(docs):
    return splitter.split_documents(docs)

