# 1. Load documents
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("new_articles", glob="*.txt")
docs = loader.load()

# 2. Split if needed
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Create Embeddings
from langchain_community.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings("your openai API key here")

# 4. Index with Chroma
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="db")
db.persist()

# 5. Vector Search
query = "Tell me about fruit nutrition"
results = db.similarity_search(query, k=3)
