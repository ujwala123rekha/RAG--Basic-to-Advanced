""" Documents
       ↓
    Loader
       ↓
    Chunking
       ↓
    Embeddings
       ↓
    Vector Database (FAISS)
       ↓
    Retriever (Top-K)
       ↓
    Prompt Template
       ↓
    LLM
       ↓
    Answer
    
    I had used 2 models for embedding and llm model

    nomic-embed-text   for embedding
    gemma 3:4b    for model
    """

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


loader = PyPDFLoader(r"C:\Users\UJWALA\Downloads\present resume 1.pdf")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)


embedding_model = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.from_documents(docs, embedding_model)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})


llm = ChatOllama(model="gemma3:4b")


prompt_template = """
Answer the question using the context below.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


def rag_pipeline(query):

    retrieved_docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(final_prompt)
    return response.content


while True:

    query = input("\nAsk something (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    answer = rag_pipeline(query)

    print("\nAnswer:\n", answer)

"""
IMPORTANT POINTS

for loading documents we use pypdfloader......for all tyopes of documents we can use UnstructureFileLoader

In normal ,we convert the loaded document in to tokens then chunks then embeddings
but using langchain recursivecharactertextsplitter we can direct convert from loaded document in to chunks
tokens : small parts of text with no meaning
|
chunks: a collection of chunks and have meaningful text

for text splitting there are many as Textsplitter,TokenTextSplitter,Sentence Splitter
out of all this we use this RecursiveCharacterTextSplitter cause it can split by para,sentence ,words and characters also

chunk_size → how large each chunk of text should be (in characters)
so each chunk has 500 characters
chunk overlap -> how much previous chunk content is repeated to maintain context continuity

tokens---embedding
first the models divides the tokens then use chunkID nad embedding matrix(each model has different )
and finalise the embedding these are used for getting the similarity btwn the words 

the embeddings can be stored in the vector store bases such as FAISS,CHROMA,PINECONE etc....

The retriever's job is to find the most relevant chunks from the vector database.
ex: we can use mmr(maximal marginal relevance) it prevents duplicate"""
