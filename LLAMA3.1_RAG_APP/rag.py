import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load the data
loader = WebBaseLoader(
    web_paths=("https://www.orfonline.org/expert-speak/the-silent-threat-exploring-the-link-between-air-pollution-and-diabetes#:~:text=Air%20pollution's%20role%20in%20diabetes,onset%20of%20type%201%20diabetes.",),
    bs_kwargs=dict(),
)
docs = loader.load()
print(docs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(splits)

# 2. Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text", num_gpu=1, show_progress=True)
print(embeddings)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
print(vectorstore)

# 3. Call Ollama Llama3 model
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='phi3:mini', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# 4. RAG Setup
retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    print("Retrieving docs")
    retrieved_docs = retriever.invoke(question)
    print("Retrieving relevant context")
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# 5. Use the RAG App
result = rag_chain("What is the relation between air pollution and type 1 diabetes?")
print(result)