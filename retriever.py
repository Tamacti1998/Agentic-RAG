from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

p_dir = "db/chroma_db"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
e_model = HuggingFaceEmbeddings(model_name = model_name)


db = Chroma(
    embedding_function = e_model,
    persist_directory = p_dir,
    collection_metadata = {"hnsw:space" : "cosine"}
)

retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 3}
)

def retrieve(query):
    """Return the 3 most relevant chunks for a user question."""
    return retriever.invoke(query)  

if __name__ == "__main__":
    query = "Can two players share a route?"
    results = retrieve(query)

    print(f"User query: {query}")
    print("-----Context------")

    for i, doc in enumerate(results, 1):
        print(f"Document {i}: \n{doc.page_content}\n")