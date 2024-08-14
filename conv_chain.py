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

# !curl https://ollama.ai/install.sh | sh

import subprocess
import time
import threading

def get_langchain_chain():
    # subprocess.run("curl https://ollama.ai/install.sh | sh", shell=True, check=True)

    # # Start the ollama server in a new process
    # process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # # Function to print server output
    # def print_output(process):
    #     while True:
    #         output = process.stdout.readline()
    #         if output == b'' and process.poll() is not None:
    #             break
    #         if output:
    #             print(output.strip().decode('utf-8'))
    #         time.sleep(1)

    # # Start a thread to print server output
    # thread = threading.Thread(target=print_output, args=(process,))
    # thread.start()

    # subprocess.run("ollama pull llama3.1:8b", shell=True, check=True)

    # print("Ollama server is running in the background")

    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"

    gpt4all_kwargs = {'allow_download': 'True'}

    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key ="iF6RFJiNfggDlFuIKvRh12Po5oTfjqXdvvOxflvU",user_agent="my-app") # type: ignore

    retriever = Chroma(persist_directory="./chroma_db", embedding_function=embeddings).as_retriever(search_kwargs={"k": 20})

    CohereRerank.update_forward_refs()

    compressor = CohereRerank(user_agent="my-app", cohere_api_key ="iF6RFJiNfggDlFuIKvRh12Po5oTfjqXdvvOxflvU", top_n=5)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )


    llm = ChatOllama(model="llama3.1:8b")

    system_prompt: str = (
        "Bạn là trợ lý trả lời văn bản luật tiếng Việt. "
        "Sử dụng ngữ cảnh được cung cấp để trả lời câu hỏi. "
        "\n\n"
        "Ngữ cảnh: \"{context}\" "
        "Bạn phải luôn trả lời từ nội dung, có thể cần phải dùng nhiều điều cho một lần trả lời, trích dẫn bắt buộc chính xác từng chữ. "
        "Nếu bạn không tìm thấy nội dung được hỏi từ ngữ cảnh, "
        "nói \"Tôi không biết\". Không được nói điều nào về \"ngữ cảnh được cung cấp\" trong câu trả lời."
    )

    contextualize_q_system_prompt = (
        "Với lịch sử hội thoại và câu hỏi mới nhất của người dùng có thể đề cập đến "
        "ngữ cảnh trong lịch sử trò chuyện, hãy tổng hợp lại thành 1 câi hỏi đôc lập "
        "mà không cần phải biết lịch sử hội thoại để hiểu được. Bạn không được trả lời câu hỏi. "
        "Chỉ cần tổng hợp lại thành 1 câu hỏi không cần lịch sử trò chuyện nếu cần, "
        "còn không thì trả về câu hỏi gốc."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        compression_retriever,
        contextualize_q_prompt
    )

    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
#            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)

#    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    rag_chain = create_retrieval_chain(compression_retriever, qa_chain)

    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    subprocess.run("curl http://localhost:11434", shell=True, check=True)

#    return conversational_rag_chain
    return rag_chain
