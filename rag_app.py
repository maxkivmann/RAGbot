import streamlit as st
from PyPDF2 import PdfReader
import langchain
import docx
import pypdf 
from textwrap import dedent
import pandas as pd
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import GooglePalm, OpenAI
from langchain_community.embeddings import GooglePalmEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import DirectoryLoader
from crewai import Agent

from tools.calculator_tools import CalculatorTools
from tools.vectordbsearch_tools import VectorSearchTools

from langchain_community.llms.ctransformers import CTransformers
from langchain_community.llms.ollama import Ollama
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import load_tools
import os


os.environ["GOOGLE_API_KEY"] = "AIzaSyD29fEos3V6S2L-AGSQgNu03GqZEIgJads" #google PaLM API Key
os.environ["OPENAI_API_KEY"] = "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" #OpenAI API Key


st.set_page_config(page_title='Personal Chatbot', page_icon='mag_right')
st.header('Knowledge Query Assistant')
st.write("I'm here to help you get information from your file.")
st.sidebar.title('Option')



def load_local_model(model_path):
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 0  # Metal set to 1 is enough.
    n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

    try:

        local_llm1 = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            temperature=0,
            n_batch=n_batch,
            n_ctx=8000,
            f16_kv=True,
            #callback_manager=callback_manager,
            verbose=True,
        )
        return local_llm1

    except ImportError:
        print("LlamaCpp not available. Falling back to CTransformers.")

    # Configuration for CTransformers
    config = {
        "max_new_tokens": 2048,
        "context_length": 8000,
        "repetition_penalty": 1.1,
        "temperature": 0,
        "top_k": 50,
        "top_p": 0.5,
        "threads": int(os.cpu_count() / 2)
    }

    from ctransformers import CTransformers

    local_llm2 = CTransformers(
        model=model_path,
        config=config,
        #callback_manager=callback_manager
    )

    return local_llm2


model_path="./models/zephyr-7b-alpha.Q4_K_M.gguf"  #local model file. add model file (.gguf) in model models folder and replace the name/path
local_model = load_local_model(model_path) #local model


ollama_llm = Ollama(model="Mistral", num_ctx=4096,temperature=0, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])) 
#install Ollama application and run with 'ollama run 'model name''. for example: ollama run llama2


google_llm = ChatGooglePalm(temperature= 0.1) #with internet
llm_openai = ChatOpenAI(temperature =0,  model='gpt-3.5-turbo-16k') #with internet

llm=local_model #you can use whichever model you want


def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    raw_text = ' '.join(allText)
    return raw_text

    
def get_csv_text(file):
    return "Empty"





@st.cache_resource(show_spinner=False)
def processing_csv_pdf_docx(uploaded_file):
    with st.spinner(text="Getting Ready"):

        # Read text from the uploaded PDF file
        data = []
        for file in uploaded_file:
            split_tup = os.path.splitext(file.name)
            file_extension = split_tup[1]
        
            if file_extension == ".pdf":

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file1:
                    tmp_file1.write(file.getvalue())
                    tmp_file_path1 = tmp_file1.name
                    loader = PyPDFLoader(file_path=tmp_file_path1)
                    documents = loader.load()
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                    data += text_splitter.split_documents(documents)


            if file_extension == ".csv":
                
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_file_path = tmp_file.name

                    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                                'delimiter': ','})
                    documents = loader.load()
                    
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        
                    data += text_splitter.split_documents(documents)
                    st.sidebar.header(f"Data-{file.name}")
                    data1 = pd.read_csv(tmp_file_path)
                    st.sidebar.dataframe(data1)
            
            if file_extension == ".docx":

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_file_path = tmp_file.name
                    loader = UnstructuredWordDocumentLoader(file_path=tmp_file_path)
                    documents = loader.load()
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

                    data += text_splitter.split_documents(documents)
                

        # Download embeddings from GooglePalm
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        #embeddings = GooglePalmEmbeddings()
        #embeddings = OpenAIEmbeddings()

        # Create a FAISS index from texts and embeddings
        vectorstore = FAISS.from_documents(data, embeddings)
        vectorstore.save_local("faiss")
        return vectorstore

@st.cache_resource(show_spinner=False)
def load_from_directory():
    with st.spinner(text="Creating Embeddings"):
        loader = DirectoryLoader("./uploaded_files")
        documents = loader.load()
        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )
        docs = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss")
        return db



with st.sidebar:
    uploaded_file =  st.file_uploader("Upload your files",
    help="Multiple Files are Supported",
    type=['pdf', 'docx', 'csv'], accept_multiple_files= True)

with st.sidebar:
    if not uploaded_file:
        st.warning("Upload your files to start chatting!")


if 'history' not in st.session_state:  
        st.session_state['history'] = []


if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"]= []
    st.session_state['history']  = []


########--Main PDF--########
def load_files():

    for file in uploaded_file:
        with open(os.path.join('uploaded_files', file.name), 'wb') as f:
            f.write(file.getbuffer())
            st.success(f'Saved file: {file.name}', icon="✅")
    processing_csv_pdf_docx(uploaded_file)
    load_from_directory() 





def main():
    try:
        
        if uploaded_file:
            load_files()
        else:
            load_from_directory()
    
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input(placeholder="Type your question!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", human_prefix= "", ai_prefix= "")

            for i in range(0, len(st.session_state.messages), 2):
                if i + 1 < len(st.session_state.messages):
                    current_message = st.session_state.messages[i]
                    next_message = st.session_state.messages[i + 1]
                    
                    current_role = current_message["role"]
                    current_content = current_message["content"]
                    
                    next_role = next_message["role"]
                    next_content = next_message["content"]
                    
                    # Concatenate role and content for context and output
                    context = f"{current_role}: {current_content}\n{next_role}: {next_content}"
                    
                    memory.save_context({"question": context}, {"output": ""})

            # Get user input -> Generate the answer
            greetings = ['Hey', 'Hello', 'hi', 'hello', 'hey', 'helloo', 'hellooo', 'g morning', 'gmorning', 'good morning', 'morning',
                        'good day', 'good afternoon', 'good evening', 'greetings', 'greeting', 'good to see you',
                        'its good seeing you', 'how are you', "how're you", 'how are you doing', "how ya doin'", 'how ya doin',
                        'how is everything', 'how is everything going', "how's everything going", 'how is you', "how's you",
                        'how are things', "how're things", 'how is it going', "how's it going", "how's it goin'", "how's it goin",
                        'how is life been treating you', "how's life been treating you", 'how have you been', "how've you been",
                        'what is up', "what's up", 'what is cracking', "what's cracking", 'what is good', "what's good",
                        'what is happening', "what's happening", 'what is new', "what's new", 'what is neww', "g’day", 'howdy']
            compliment = ['thank you', 'thanks', 'thanks a lot', 'thanks a bunch', 'great', 'ok', 'okay', 'great', 'awesome', 'nice']
                
        
                        
            prompt_template =dedent(r"""
            You are a helpful assistant to help students study from their study material.
            talk humbly. Answer my question from the provided context. Do not answer from your own training data.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know. Do not makeup any answer.
            Do not answer hypothetically.
            Please Do Not say: "Based on the provided context"
            Always use the context to find the answer.
            
            
            this is the context from study material:
            ---------
            {context}
            ---------

            This is chat history between you and user: 
            ---------
            {chat_history}
            ---------

            New Question: {question}

            Helpful Answer: 
            """)
            embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'})
            db = FAISS.load_local("faiss", embeddings)


            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question", "chat_history"]
            )

            # Run the question-answering chain
            docs = db.similarity_search(prompt, k=5)
            

                # Load question-answering chain
            chain = load_qa_chain(llm=llm, verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
                
            #chain = load_qa_chain(ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613", streaming=True) , verbose= True, prompt = PROMPT, memory=memory,chain_type="stuff")

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                if prompt.lower() in greetings:
                    response = 'Hi, how are you? I am here to help you get information from your file. How can I assist you?'
                    st.session_state.messages.append({"role": "Assistant", "content": response})
                elif prompt.lower() in compliment:
                    response = 'My pleasure! If you have any more questions, feel free to ask.'
                    st.session_state.messages.append({"role": "Assistant", "content": response})
                else:
                    with st.spinner('Bot is typing ...'):
                
                        response = chain.run(input_documents=docs, question = prompt)#, callbacks=[st_cb])
                        st.session_state.messages.append({"role": "Assistant", "content": response})
                        
                st.write(response)
    except Exception as e:
        result = "Sorry, I do not know the answer to this question "
        return result


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


if __name__ == '__main__':
    main()

