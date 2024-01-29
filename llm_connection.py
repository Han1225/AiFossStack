
import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.llms import Ollama
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#### This app lets you to ask questions from your own documents using a local LLM##

####### Default llm configuration ##########
default_index = "index"
default_top_k = 50 # k:top n chuncks
default_temperature = 0.9 # temperature: 0 (not creative) - 1 (creative)
default_model = "stablelm-zephyr"
default_search_type = "similarity" 
default_question = "Write a speech about..."
####### Defaults for indexing new content ##
default_data_folder = "/Users/hann/Downloads/test/indexes" # specify the folder to store new indexes
default_indexes_folder = "indexes/"
default_chunk_size = 1000
default_overlap = 200

####### Default System Template ##
# Improvements: Show it on the website and change
template = """You are an expert. Write following these criteria:

* Cite the exact source next to each paragraph.
* Indicate date, location, and entities related to each fact you cite.
* Write in an elegant, professional, diplomatic style.
Answer the question based only on the following context:
{context}
If there is not enough information say 'I don't have enough information'.
Question: {question}

"""


####### Helper functions ##
#@st.cache_data
def load_index(index_name):
    try:
        embeddings=OllamaEmbeddings()
        vectorstore = FAISS.load_local(index_name , embeddings)
        status = {"status" : "success loading index"}
        return vectorstore
    except:
        status = {"status" : "error loading index"}
        return status
    finally:
        print (status)


def rag_query(index_name, question, template, model="stablelm-zephyr", temperature = 0.9, search_type="similarity", k=50):
    # chain elements
    vectorstore = load_index(index_name)
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k":k}, return_source_documents=True)
    prompt = ChatPromptTemplate.from_template(template)
    model = Ollama(model=model, temperature = temperature)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | model | StrOutputParser()
    )
    result = chain.invoke(question)
    return result

# Get a list of 'topics' in the directory


def index_from_folder(input_dir, index_name, chunk_size=1000, chunk_overlap=200):
    try:       
        data = DirectoryLoader(input_dir , use_multithreading=True, loader_cls=TextLoader).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(data)
        db = FAISS.from_documents(documents=all_splits, embedding=OllamaEmbeddings())
        db.save_local(index_name)
        status = {"files_ingested" : len(data) , "chunks_created" : len(all_splits), "index_created" : index_name}   
        print (status)    
    except:
        status = {"error" : "indexing error"}
    finally:
        print (status) 
        return status
        
########### Web app Streamlit ########
st.set_page_config(layout="wide" , initial_sidebar_state="collapsed")
#### Sidebar
with st.sidebar:
    st.write("## Settings:")
    st.write("Default index: ", default_index)
    with st.expander("## Create a new index", expanded=False):
        with st.form("NewIndex"):
            st.write("## Create a new index:")
            input_dir = st.text_input("From folder (e.g. ./data/): ", value = default_data_folder)
            folder_names = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
            folder_index = st.selectbox('Choose a folder: ', folder_names)
            # Initialize or get the current state
            index_name = st.text_input("Name: ", value = 'index')
            chunk_size = st.number_input('Chunk size' , value = default_chunk_size)
            chunk_overlap = st.number_input('Chunk overlap' , value = default_overlap)
            # submit button
            submitted = st.form_submit_button("Create")
            if submitted:
                indexing_status = index_from_folder(input_dir+folder_index, default_indexes_folder + index_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.write(indexing_status)
                
                # improvement: add references 
     
########## Main page area ########
col1, col2 = st.columns(2)

col1.image("FordhamLogo.png", width=400)
col2.image("UNtechLogo.png", width=400)
st.subheader("This is your speech writing assistant based on your content. I respect your privacy. You data will never leave your computer.")



####### Query form
with st.form("Query"):
    with st.expander('Expand to select a folder'):
        # Update folder names based on the input directory
        folder_names = [name for name in os.listdir(input_dir+default_indexes_folder) if os.path.isdir(os.path.join(input_dir+default_indexes_folder, name))]
        folder_index = st.selectbox('Choose an index: ', folder_names)
        
    user_question = st.text_area("Task me!", value = default_question)
    # submit button
    submitted = st.form_submit_button("Go")
    if submitted:
        result = rag_query(default_indexes_folder + folder_index, user_question, template, model=default_model, temperature = default_temperature, search_type=default_search_type, k=default_top_k)
        st.write(result)

#### NEXT STEPS
# Add sources
# Use minimum similarity score
# Dynamic settings sidebar to change configuration
