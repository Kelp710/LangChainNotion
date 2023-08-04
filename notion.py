import os
from psychicapi import Psychic
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.llms import Cohere
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
# from langchain.llms import AzureOpenAI
import logging
load_dotenv(".env")

try:
    # os.environ["OPENAI_API_TYPE"] = "azure"
    # os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
    # os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    # os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    PSYCHIC_SECRET_KEY = os.getenv("PSYCHIC_SECRET_KEY")
    ACCOUNT_ID = os.getenv("ACCOUNT_ID")
    OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # COHERE_API_KEY=os.getenv("COHERE_API_KEY")

    print("Syncing your docs through Psychic...")
    psychic = Psychic(secret_key=PSYCHIC_SECRET_KEY)
    raw_docs = psychic.get_documents(account_id=ACCOUNT_ID).documents
    if raw_docs is None:
        raise Exception("No docs found!")
    print(
    "Generating embeddings from your docs and inserting them into Chroma...")
    documents = [
    Document(
        page_content=doc["content"],
        metadata={
        "title": doc["title"],
        "source": doc["uri"]
        },
    ) for doc in raw_docs
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                    chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=OpenAI_API_KEY)
    # embeddings = CohereEmbeddings(model="embed-english-light-v2.0", cohere_api_key=COHERE_API_KEY)
    vdb = Chroma.from_documents(texts, embeddings)

    # llm = AzureOpenAI(
    #     deployment_name="gpt-09",
    #     model_name="gpt-4",
    # )
    llm=OpenAI(temperature=0, openai_api_key=OpenAI_API_KEY, model_name="gpt-4")

    # llm=OpenAI()
    # llm=Cohere(cohere_api_key=COHERE_API_KEY,model="command")
    chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vdb.as_retriever())
    print("Document embeddings loaded.")
    while True:
        QUERY = input("✨ Ask a question: ")
        answer = chain({"question": QUERY}, return_only_outputs=True)
        print("Answer: "+answer["answer"])
        print("Sources:"+answer["sources"])
except Exception as e:
  print("Encountered an error: ", e)
