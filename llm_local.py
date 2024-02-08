from flask import Flask, request, jsonify

# from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# from langchain.llms import HuggingFaceHub
import os
from langchain_community.llms import LlamaCpp
import boto3
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub


# from langchain_community.llms import LlamaCpp
# from fpdf import FPDF


os.environ["AWS_ACCESS_KEY_ID"] = "AKIAXHRYJD7GZDMNXQFK"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"


# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
app = Flask(__name__)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.route("/apiQues", methods=["GET"])
def local_llm():
    question = request.args.get("question")
    current_directory = os.getcwd()
    filename = "caseDoc.pdf"
    file_path = os.path.join(current_directory, filename)
    loader = PyPDFLoader(file_path)
    data = loader.load()
    # documents = data
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = FAISS.from_documents(
        documents=all_splits, embedding=GPT4AllEmbeddings()
    )

    # n_gpu_layers = 1  # Metal set to 1 is enough.
    # n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        # model_url="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_0.gguf",
        model_path="F:\Documents backup\AI Projects\llm_rag\llama-2-7b-chat.Q5_0.gguf",
        # n_gpu_layers=n_gpu_layers,
        # n_batch=n_batch,
        n_ctx=4096,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        verbose=True,
    )

    # Prompt
    rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
    print(rag_prompt_llama)
    # rag_prompt_llama.messages
    retriever = vectorstore.as_retriever()
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_llama
        | llm
        | StrOutputParser()
    )
    return qa_chain.invoke(question)


@app.route("/apiCaseId", methods=["GET"])
def documentDownload():
    caseId = request.args.get("caseId")
    s3 = boto3.client("s3")
    bucket_name = "wilegaldocs"
    # List objects in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name)
    # print(respnse)
    # Iterate through the objects and filter based on case number
    for obj in response["Contents"]:
        # Extract the case number from the object key
        key_parts = obj["Key"].split("_")
        if len(key_parts) >= 2 and key_parts[1] == caseId:
            # Download the object to your server
            s3.download_file(bucket_name, obj["Key"], "caseDoc.pdf")
            current_directory = os.getcwd()
            filename = "caseDoc.pdf"
            file_path = os.path.join(current_directory, filename)
            jsonData = {"path": file_path}
            return jsonify(jsonData)
        else:
            return jsonify({"msg": "No data found"})


def loading_dataKnowledge():
    # Load data from PDF and create vector representations
    current_directory = os.getcwd()
    filename = "caseDoc.pdf"
    file_path = os.path.join(current_directory, filename)
    loader = PyPDFLoader(file_path)
    data = loader.load()
    documents = data
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Set up embeddings and create vector store
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    # Set up language model for question answering
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.7, "max_length": 512},
    )
    chain = load_qa_chain(llm, chain_type="stuff")

    return db, chain


@app.route("/delete", methods=["GET"])
def filedelete():
    current_directory = os.getcwd()
    filename = "caseDoc.pdf"
    file_path = os.path.join(current_directory, filename)
    os.remove(file_path)
    return jsonify({"msg": "File Deleted"})


def response(msg, db):
    # Define your response processing logic here
    docs = db.similarity_search(msg)
    return docs


@app.route("/api", methods=["GET"])
def get_bot_response():
    # Get the question from the request
    msg = request.args.get("question")

    # Load data and initialize question answering chain
    db, chain = loading_dataKnowledge()

    # Get response from the question answering chain
    bot_response = chain.run(input_documents=response(msg, db), question=msg)

    # Extract relevant information
    start_index = bot_response.find("Helpful Answer:")
    if start_index != -1:
        relevant_info = bot_response[start_index + len("Helpful Answer:") :].strip()
    else:
        relevant_info = "No relevant information found."

    return jsonify({"Answer": relevant_info})


if __name__ == "__main__":
    app.run(port=8080)
