import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from config import file_path


load_dotenv()

# function that loads the files
def load_doc(file_path):
    # Check the existence of the folder
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The folder {file_path} doesn't exist.")

    # Load the files from the folder
    files = DirectoryLoader(
        path = file_path,
        glob = "*.txt",
        loader_cls = TextLoader
    )

    loaded_files = files.load()

    if len(loaded_files) == 0:
        raise FileNotFoundError(f"No files presents in {file_path}")

    for i, f in enumerate(loaded_files[:4]):
        print(f"\nDocument {i+1}:")
        print(f" Source: {f.metadata['source']}")
        print(f" Content length: {len(f.page_content)} chaaracters")
        print(f" Content preview: doc {f.page_content[:150]}..")
        print(f" metadata: {f.metadata}")
       

    return loaded_files


# function for chunking
def chunking(doc, chunk_size = 300, overlap = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = overlap,
        separators = ["\n\n", "\n", ". "," ", ""]
    )

    chunks = splitter.split_documents(doc)

    if chunks:
            for i, c in enumerate(chunks[:3]):
                print(f"\n---- Chunk {i+1}----")
                print(f" Source: {c.metadata['source']}")
                print(f" Contenet length: {len(c.page_content)} chaaracters")
                print(f" Content")
                print(c.page_content)
                print("-" * 50)
    
            if len(chunks) > 5:
                print(f"\n... and {len(chunks) - 5} more chunks.")

    return chunks

# embed the chunks and store in vector database
def embed_store(chunks):
    e_model = "sentence-transformers/all-MiniLM-L6-v2"

    embedding_model = HuggingFaceEmbeddings(
        model_name = e_model
    )

    p_dir = "db/chroma_db"

    vdb = Chroma.from_documents(
        embedding = embedding_model,
        documents = chunks,
        persist_directory = p_dir,
        collection_metadata = {"hnsw:space" : "cosine"}
    )

    print("------Finished Creating Vector DB--------")

    print(f"Database Storage created at: {p_dir}")

    return vdb


def main():
    # Rebuild the local index so repeated ingestion runs do not create duplicates.
    if os.path.exists("db/chroma_db"):
        shutil.rmtree("db/chroma_db")

    # for loading the files
    f = load_doc(file_path)

    # chunks
    c = chunking(f)

    # Embed and Store
    es = embed_store(c)


if __name__ == "__main__":
    main()

    