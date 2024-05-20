from llama_index.core import StorageContext, ServiceContext, load_index_from_storage
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
import chainlit as cl

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@cl.on_chat_start
async def factory():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    embed_model = GeminiEmbedding(
        model_name="models/embedding-001", api_key=GOOGLE_API_KEY
    ) 

    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

    service_context = ServiceContext.from_defaults(
                        embed_model=embed_model, llm=llm,
                        callback_manager=
                        CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    index = load_index_from_storage(
                storage_context, 
                service_context=service_context
    )

    chat_engine = index.as_chat_engine(
        service_context=service_context,
        similarity_top_k=10
    )

    cl.user_session.set("chat_engine", chat_engine)

@cl.on_message
async def main(message: cl.Message):
    chat_engine = cl.user_session.get("chat_engine")  
    response = await cl.make_async(chat_engine.query)(message.content)

    response_message = cl.Message(content="")

    for token in response.response:
        await response_message.stream_token(token=token)

    await response_message.send()
	
#“pip install chainlit” to create the above file. There is a default config.toml file (found under .chainlit folder) along with the application which serves the 
#Readme for the app, can be configured to give custom HTML, CSS, and JS codes to make changes to the look and feel of the app along with other app-related settings 
#like telemetry.

#To run the above code from the terminal save it into the app.py file and run the command “chainlit run app.py”.

#Refer to official docs: https://docs.chainlit.io/integrations/llama-index