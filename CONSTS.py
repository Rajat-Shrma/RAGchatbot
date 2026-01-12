from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

# model = ChatGoogleGenerativeAI(model = 'models/gemini-2.5-flash')
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.2
)

model = ChatHuggingFace(llm=llm)

embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',

)


from langchain_chroma import Chroma

vectorstore = Chroma(
        collection_name= 'my_collection',
        embedding_function = embedding_model,
        persist_directory ='chroma_db'
    )

retriever = vectorstore.as_retriever(search_kwargs = {"k":1})


