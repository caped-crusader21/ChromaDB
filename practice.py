from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  # Langchain supports this
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your text file
with open("new_articles.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(text)

# Use a local sentence-transformers model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # downloads locally

# Create vector store
vectordb = Chroma.from_texts(texts, embeddings, collection_name="microsoft_news")

# Query
query = "Microsoft cloud revenue growth"
results = vectordb.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"Result {i+1}:\n{doc.page_content}\n")
