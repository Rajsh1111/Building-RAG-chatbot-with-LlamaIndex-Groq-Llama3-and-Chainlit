pip install llama-index
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    ServiceContext, 
    load_index_from_storage
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq

reader = SimpleDirectoryReader(input_dir="path/to/directory")
documents = reader.load_data()

#for multi-processing:
documents = reader.load_data(num_workers=4)

#Llamaindex offers a bundle of embedding model options ranging from both open & closed-source models.

pip install llama-index-embeddings-huggingface llama-index-embeddings-gemini

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

pip install google-generativeai>=0.3.0 llama-index-embeddings-gemini

from llama_index.embeddings.gemini import GeminiEmbedding
embed_model = GeminiEmbedding(model_name="models/embedding-001")

#For this particular RAG pipeline, we’ve made use of Google Gemini Embeddings having an embedding dimension of 768. Acquire the API key from Google Makersuite, 
#wherein we can sign in using our Google account and create a new key.

#model_name defaults to 'models/embedding-001'

#Types of chunking:

#Character Splitting — Simple static character chunks of data
#Recursive Character Text Splitting — Recursive chunking based on a list of separators
#Document Specific Splitting — Various chunking methods for different document types (PDF, Python, Markdown)
#Semantic Splitting — Embedding walk-based chunking

splitter = SemanticSplitterNodeParser(
              buffer_size=1, 
              breakpoint_percentile_threshold=95, 
              embed_model=embed_model
           )

nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

#Transformations & Ingestion Pipeline

#After the data is loaded, you then need to process and transform your data before putting it into a storage system. These transformations include chunking, 
#extracting metadata, and embedding each chunk. This is necessary to make sure that the data can be retrieved and used optimally by the LLM.

#Transformation input/outputs are Node objects (a Document is a subclass of a Node). Transformations can also be stacked and reordered.

from llama_index.core import Document
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
        GeminiEmbedding(),
    ]
)

# run the pipeline
nodes = pipeline.run(documents=[Document.example()])

#An IngestionPipeline uses a concept Transformations that is applied to input data. These Transformations are applied to input data, and the resulting nodes are 
#either returned or inserted into a vector database (if given). Each node+transformation pair is cached so that subsequent runs (if the cache is persisted) with the 
#same node+transformation combination can use the cached result and save you time.

#The Llamaindex Settings module helps in configuring global settings that can be used throughout the pipeline. Earlier the same could be achieved via ServiceContext.

from llama_index.llms.groq import Groq
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

Settings.llm = Groq(model="llama3-70b-8192")
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
Settings.text_splitter = SentenceSplitter(chunk_size=1024)
Settings.chunk_size = 512
Settings.chunk_overlap = 20
Settings.transformations = [SentenceSplitter(chunk_size=1024)]
# maximum input size to the LLM
Settings.context_window = 4096

# number of tokens reserved for text generation.
Settings.num_output = 256

#Vector Store Index
#After the data is loaded, chunked and embedded its time to now store it in a vector database. For this example, we’re going to make use of LlamaIndex in-memory db 
#called vector store index. This will create a bunch of JSON files with vector embedding chunks once persisted from primary to secondary memory storage using the 
#StorageContext module.
#Vector stores accept a list of Node objects and build an index from them.

vector_index = VectorStoreIndex.from_documents(
                  documents, show_progress=True, 
                  service_context=service_context, 
                  node_parser=nodes
               )

vector_index.storage_context.persist(persist_dir="./storage")

storage_context = StorageContext.from_defaults(persist_dir="./storage")

index = load_index_from_storage(
            storage_context, 
            service_context=service_context
        )
#After the storage is created & persisted we can make use of it in future calls, with the load_index_from_storage() method which accepts the service context 
#object and storage_context object.

#Open-Source LLMs with Groq
#Groq LPU (Language Processing Engine) has been revolutionizing the Open Source LLM space. It has a deterministic, single-core streaming architecture that sets the 
#standard for GenAI inference speed with predictable and repeatable performance for any given workload.

pip install llama-index-llms-groq
llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

#GroqCloud API makes it to any developer’s fingertips to easily plug into the LLMs without hosting locally, directly getting the powerhouse with the least latency 
#rates recorded to date!

#Signup/log in to GroqCloud an account and get started by creating an API key at https://console.groq.com/keys.

#Meta developed and released the Meta Llama 3 family of large language models (LLMs), a collection of pre-trained and instruction-tuned generative text models in 
#8 Billion and 70 Billion sizes. The Llama 3 instruction-tuned models are optimized for dialogue use cases and outperform many available open-source chat models on 
#common industry benchmarks.

#Query/Chat Engine
#The above-created index variable can now be referenced to query over user questions and build a QnA system using the as_query_engine() method. The same can be 
#turned into a chatbot having context and chat history for the session using the chat_engine = as_chat_engine() function with the same set of parameters and would 
#change to chat_engine.chat(query). This we will see in the next segment while building the final chatbot.

query_engine = index.as_query_engine(
                  service_context=service_context,
                  similarity_top_k=10,
                )

query = "What is difference between double top and double bottom pattern?"
resp = query_engine.query(query)

print(resp.response)

#A double top pattern is a bearish technical reversal pattern that forms after an asset reaches a high price twice with a moderate decline between the two highs, 
#and is confirmed when the asset's price falls below a support level equal to the low between the two prior highs. On the other hand, a double bottom pattern is a
#bullish reversal pattern that occurs at the bottom of a downtrend, signaling that the sellers are losing momentum and resembles the letter “W” due to the 
#two-touched low and a change in the trend direction from a downtrend to an uptrend. In summary, the key difference lies in the direction of the trend change and 
#the shape of the pattern.

#The above answer is from the document fed “All Chart Patterns.pdf” file which has the context of different stock market-related candlestick charts to showcase.

#Now that our RAG pipeline is built, we can serve it as a SaaS model to end customers with the help of a chatbot. To build this, we’ll use the Chainlit library 
#which is built on top of Streamlit, the popular Python app to showcase quick Data Science and ML demos. Chainlit has a UI similar to that of ChatGPT.
#Pls. refer to app.py file in this repo for the same.