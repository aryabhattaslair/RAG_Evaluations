import os
from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as pine # type: ignore
import openai
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import FileCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import asyncio
from loguru import logger
#from holdingAnalysis import read_json_and_extract
import config

STORE = {}

def load_docs(dataset_name, col):
    loader = HuggingFaceDatasetLoader(path=dataset_name, page_content_column=col)
    # documents = loader.load()
    documents = loader.load()[:20]
    print(documents)
    return documents

def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in STORE:
        STORE[session_id] = ChatMessageHistory()
    return STORE[session_id]

def create_backend(build_index=False, docs=None):

    vector_store = None

    secrets = config.Secrets()

    pinecone_api_key = secrets.pinecone_api_key
    openai_api_key = secrets.openai_api_key

    print("pinecone_api_key = ", pinecone_api_key)
    print("openai_api_key = ", openai_api_key)

    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["PINECONE_INDEX_NAME"] = "rag-news-03122023"

    pc = Pinecone(api_key=pinecone_api_key)

    environment = "gcp-starter"
    index_name = "rag-news-03122023"

    llm = ChatOpenAI(openai_api_key=openai_api_key,model="gpt-3.5-turbo", temperature=0)
    
    index = pc.Index(index_name)
    print('index = ', index)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)

    if build_index and docs is not None:
        texts = [doc.page_content for doc in docs]
        embedding_result = embeddings.aembed_documents(texts=texts)
        # print("embedding_result type= ", type(embedding_result))

        vectors = []
        for i in range(len(embedding_result)):
            vector = {
                "id": str(i),
                "values": embedding_result[i],
                # "metadata": docs[i].metadata
                "metadata": {'text': docs[i].page_content}
            }
            vectors.append(vector)

        max_chunk_size = 1000

        for i in range(0, len(vectors), max_chunk_size):
            print("we are here bro !!!!")
            chunk = vectors[i:i+max_chunk_size]
            
            try:
                index.upsert(vectors=chunk)
                print(f"Upserted chunk {i//max_chunk_size + 1}/{len(vectors)//max_chunk_size + 1}")
            except Exception as e:
                print(f"Error upserting chunk {i//max_chunk_size + 1}/{len(vectors)//max_chunk_size + 1}: {e}")


        text_field = "text"
        vector_store = pine(index, embeddings.embed_query, text_field, namespace="fdf")
        print("vector_store = ", vector_store)
    else:
        searcher = PineconeVectorStore(pinecone_api_key=pinecone_api_key, index_name="rag-news-03122023", embedding=embeddings)
        retriever = searcher.as_retriever()
        # retriever.search_kwargs['maximal_marginal_relevance'] = True
        retriever.search_kwargs['k'] = 3
        # vector_store = pine(index, embeddings.embed_query, "text", namespace="fdf")

    return llm, embeddings, retriever

# def get_stock_holdings():
#   json_file_path = "holdings.json"  # Replace with the actual path to your holdings JSON file
#    stock_holdings = read_json_and_extract(json_file_path)
#    return stock_holdings

def converse(docs=None, chain_type='stuff'):
    logfile = "../logs/rag.log"
    logger.add(logfile, colorize=True, enqueue=True)
    handler = FileCallbackHandler(logfile)
    
    if docs is None:
        llm, _, retriever = create_backend()
    else:
        llm, _, vector_store = create_backend(build_index=True, docs=docs)
        print("llm = ", llm)
        print("vector_store = ", vector_store)

    
    '''
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>
        {context}
        </context>

        Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    # retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": "Who fired Sam Altman?"})
    print(response["answer"])
    '''
    
    prompt_search = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])

    search_query_chain = create_history_aware_retriever(llm, retriever, prompt_search)
    
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

    # chat_history = [HumanMessage(content="Was Sam Altman Fired?"), AIMessage(content="Yes!")]
    # temp = conversation_chain.invoke({
    #        "chat_history": chat_history,
    #        "input": "Tell me who fired him?"
    #    })
    
    #chat_history = []#[c]
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
        memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True, 
        output_key='answer')
    
    chatQA = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}), 
        chain_type=chain_type, 
        memory=memory,
        callbacks=[handler])
    
    print("memory = ", memory)
    print("chatQA = ", chatQA)
    
    chat_history = []
    qry = ""
    while qry != 'done':
        qry = input('Question: ')
        response = chatQA({"question": qry, "chat_history": chat_history})
        print("response = ", response)
        print('****************************************')
        chat_history.append((qry, response["answer"]))
    '''


if __name__ == "__main__":
    #hf_dataset_name = "godivyam/business-companies-news-dataset"
    #column_name = "Content"

    #documents = load_docs(hf_dataset_name, column_name)
    #print("Total number of documents:", len(documents))

    #docs = split_docs(documents)
    #print("Number of document chunks:", len(docs))
    docs = None

    converse(docs, chain_type="stuff")
    # asyncio.run(converse(None, chain_type="map_reduce"))
