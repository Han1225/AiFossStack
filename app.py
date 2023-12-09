
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.llms import Ollama
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#### This app lets you to ask questions from your own documents using a local LLM##

####### Default llm configuration ##########
default_index = "index"
default_top_k = 50
default_temperature = 0.9
default_model = "mistral-openorca"
default_search_type = "similarity"
default_question = "Describe contents of my context documents?"
####### Defaults for indexing new content ##
default_data_folder = "./data/"
default_indexes_folder = "indexes/"
default_chunk_size = 1000
default_overlap = 200

####### Default System Template ##
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

def rag_query(index_name, question, template, model="mistral-openorca", temperature = 0.9, search_type="similarity", k=50):
    # chain elements
    vectorstore = load_index(index_name)
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k":k}, return_source_documents=True)
    prompt = ChatPromptTemplate.from_template(template)
    model = Ollama(model=model, temperature = temperature )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | model | StrOutputParser()
    )
    result = chain.invoke(question)
    return result

def index_from_folder(input_dir, index_name, chunk_size=1000, chunk_overlap=200):
    try:
        data = DirectoryLoader(input_dir , use_multithreading=True).load()
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
            index_name = st.text_input("Name: ", value = "index")
            chunk_size = st.number_input('Chunk size' , value = default_chunk_size)
            chunk_overlap = st.number_input('Chunk overlap' , value = default_overlap)
            # submit button
            submitted = st.form_submit_button("Create")
            if submitted:
                indexing_status = index_from_folder(input_dir, default_indexes_folder + index_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.write(indexing_status)
     
########## Main page area ########
st.image("logo.png", width=400)
st.title("Your content + your local AI = your privacy.")

####### Query form
with st.form("Query"):
    user_question = st.text_area("Task me!", value = default_question)
    # submit button
    submitted = st.form_submit_button("Go")
    if submitted:
        result = rag_query(default_indexes_folder + default_index, user_question, template, model=default_model, temperature = default_temperature, search_type=default_search_type, k=default_top_k)
        st.write(result)

#### NEXT STEPS
# Add sources
# Use minimum similarity score
# Dynamic settings sidebar to change configuration