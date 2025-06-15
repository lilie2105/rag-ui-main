import yaml
from datetime import datetime

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.llms import ChatMessage



CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200


def read_config(file_path):
    """Read YAML configuration file."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return None


config = read_config("secrets/config.yaml")


# Initialize LLM and embedder using Azure OpenAI settings
llm = AzureOpenAI(
    model=config["chat"]["azure_deployment"],
    deployment_name=config["chat"]["azure_deployment"],
    api_key=config["chat"]["azure_api_key"],
    azure_endpoint=config["chat"]["azure_endpoint"],
    api_version=config["chat"]["azure_api_version"],
)

embedder = AzureOpenAIEmbedding(
    model=config["embedding"]["azure_deployment"],  # same as deployment for now
    deployment_name=config["embedding"]["azure_deployment"],
    api_key=config["embedding"]["azure_api_key"],
    azure_endpoint=config["embedding"]["azure_endpoint"],
    api_version=config["embedding"]["azure_api_version"],
)

# Set global Settings for llama_index
Settings.llm = llm
Settings.embed_model = embedder

vector_store = SimpleVectorStore()


def store_pdf_file(file_path: str, doc_name: str):
    """Load a PDF file, split into chunks, embed and store in the vector store.

    Args:
        file_path (str): Path to the PDF file.
        doc_name (str): Name to associate with the document metadata.
    """
    loader = PyMuPDFReader()
    documents = loader.load(file_path)

    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE)
    text_chunks = []
    doc_idxs = []

    for doc_idx, doc in enumerate(documents):
        chunks = text_parser.split_text(doc.text)
        text_chunks.extend(chunks)
        doc_idxs.extend([doc_idx] * len(chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        # Add source document metadata (including doc_name)
        node.metadata = src_doc.metadata
        # Ensure document name is included in metadata for tracking
        node.metadata['document_name'] = doc_name
        node.metadata['insert_date'] = datetime.now()
        nodes.append(node)

    # Generate embeddings for each node and assign
    for node in nodes:
        embedding = embedder.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = embedding

    vector_store.add(nodes)


def delete_file_from_store(name: str) -> int:
    """Not implemented for llama_index vector store."""
    raise NotImplementedError('Function not implemented for LlamaIndex vector store.')


def inspect_vector_store(top_n: int = 10) -> list:
    """Not implemented for llama_index vector store."""
    raise NotImplementedError('Function not implemented for LlamaIndex vector store.')


def get_vector_store_info():
    """Not implemented for llama_index vector store."""
    raise NotImplementedError('Function not implemented for LlamaIndex vector store.')


def retrieve(question: str, k: int = 5):
    query_embedding = embedder.get_query_embedding(question)
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=k,
        mode="default"
    )
    query_result = vector_store.query(vector_store_query)
    if query_result.nodes is None:
        return []
    return query_result.nodes


def build_qa_messages(question: str, context: str, language: str = "français") -> list:
    system_prompt = (
        f"You are an assistant that answers questions in {language}.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, just say that you don't know.\n"
        "Use three sentences maximum and keep the answer concise.\n"
        f"{context}"
    )
    return [
        ChatMessage(role="system", content="You are an assistant for question-answering tasks."),
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=question),
    ]

def answer_question(question: str, language: str = "français", k: int = 5) -> str:
    docs = retrieve(question, k=k)

    if not docs:
        return "Je suis désolé, je n'ai trouvé aucun document pertinent pour répondre à votre question."

    docs_content = "\n\n".join(doc.get_content() for doc in docs)

    messages = build_qa_messages(question, docs_content, language=language)
    response = llm.chat(messages)
    return response.message.content

