from langchain_community.document_loaders import TextLoader, DirectoryLoader

from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.retrieval import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers.document_compressors.cohere_rerank import CohereRerank

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

loader = DirectoryLoader(
    "./articles",
    glob="*.txt",
    show_progress=True,
    use_multithreading=True,
    loader_cls=TextLoader
)

data = loader.load()

print(f"Loaded: {len(data)} document(s)")

model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
# gpt4all_kwargs = {'allow_download': 'True'}
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key ="iF6RFJiNfggDlFuIKvRh12Po5oTfjqXdvvOxflvU",user_agent="my-app")

db = Chroma.from_documents(
    documents=data,
    embedding=embeddings
    ,persist_directory="./chroma_db"
)
