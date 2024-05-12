# importing libraries
import os
import openai
import langchain
import chromadb

from langchain_community.document_loaders import OnlinePDFLoader, UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import config

# from langchain.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
# from langchain.llms import OpenAI
# from langchain.vectorstores import Chroma
# from langchain.chains.question_answering import load_qa_chain

STORE = {}

def create_embedding(openai_key, pdf_file):
    loader = PyPDFLoader(pdf_file)
    pdfData = loader.load()

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    splitData = text_splitter.split_documents(pdfData)


    collection_name = "attention_collection"
    local_directory = "attn_vect_embedding"
    persist_directory = os.path.join("data", local_directory)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectDB = Chroma.from_documents(splitData,
                        embeddings,
                        collection_name=collection_name,
                        persist_directory=persist_directory
                        )
    vectDB.persist()
    return vectDB

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in STORE:
        STORE[session_id] = ChatMessageHistory()
    return STORE[session_id]

def main():

    secrets = config.Secrets()

    os.environ["OPENAI_API_KEY"] = secrets.openai_api_key
    openai_key=os.environ.get('OPENAI_API_KEY')

    pdf_file = secrets.pdf_file
    vector_db = create_embedding(openai_key=openai_key, pdf_file=pdf_file)

    # A flow to run with Chat History. For a Chat Application
    prompt_search = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])

    llm = ChatOpenAI(openai_api_key=openai_key,model="gpt-3.5-turbo", temperature=0)
    search_query_chain = create_history_aware_retriever(llm, vector_db.as_retriever(), prompt_search)
    
    prompt_chat = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt_chat)
    conversation_chain = create_retrieval_chain(search_query_chain, document_chain)

    conversation_chain = RunnableWithMessageHistory(
        conversation_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    qry = ""
    while qry != 'done':
        qry = input('Question: ')
        response = conversation_chain.invoke(
        {"input": qry},
            config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
        )["answer"]
        print(response)
        print('****************************************')

    '''
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chatQA = ConversationalRetrievalChain.from_llm(
            OpenAI(openai_api_key=openai_key,
               temperature=0, model_name="gpt-3.5-turbo"), 
            vector_db.as_retriever(),
            memory=memory)
    
    chat_history = []
    qry = ""
    while qry != 'done':
        qry = input('Question: ')
        if qry != exit:
            response = chatQA({"question": qry, "chat_history": chat_history})
            print(response["answer"])
    '''

if __name__ == "__main__":

    main()