# The Artificial Intelligence Free & Open-Source Stack (AI FOSS Stack)

The objective of the AI FOSS Stack is to serve as a foundational building block to build apps that leverage pre-trained large language models (LLMs) which are open source and can run on low-specification hardware.

This solution is developed in the context of a collaboration between Fordham University and the UN Office of Information and Communications Technology. Read more: (https://idea.unite.un.org/servlet/hype/IMT?userAction=Browse&templateName=&documentId=9a2ac7887125130f9c0b845f27418e49)

The team includes Hannah, Remi, Peter, Sophie, Robert, and more...

# Installation (For Linux or Mac):

## Language Model
* Install a local LLM. Go to Ollama.ai to install and run your local LLM.
* Download the zephyr model using Ollama. From a terminal window type "$ollama run zephyr". If your computer becomes slow you can try using an LLM with a smaller footprint such as orca-mini using "$ollama run orca-mini"

## Obtain the app code
* $git clone http://this_Repo_URL.git

## Prep the environment
* $cd [path_to your_new_local_repo_clone_folder]
* create virtual env $python -m venv .venv
* activate it $source .venv/bin/activate     
* Data: place the knowledge files (text or pdf) that you wish to use as source inside the "data" folder. You can create subfolders. The repo contains a pdf version of the UN Charter for demo, feel free to replace it by your own docs.

## Install required packages (Internet needed)
* $pip install --upgrade pip
* $pip install -r requirements.txt

## Run the app
* $streamlit run proto_aifoss.py --server.port 8504
* Your app will be running at http://localhost:8504
* NOTE: the first time you open the app it might take a few minutes to finish loading because some downloads happen in the background. After this initial download you can run the app offline. 


## Current Features
* Retrieval-augmented Generation (RAG) based on data stored in a local folder
* Generate sentence embeddings locally
* Generate text responses using Ollama Local LLM
* The app works offline (after installation with a internet connection)

## Upcoming Features
* Persist local collection of indexed documents. (Currently the database of embeddings of the source documents is currently created at load time. This is impractical for large collections of docs.)
* Configuration page to set key variables e.g: language model, embedding library, data sources folder, chat mode, temperature, system prompt, etc.
* Avoid hallucinations. Ensure bot gives answers based exclusively on source materials. (If there is not enough info in the source materials, say "I do not have enough information").
* Create and manage various collections of indexed source documents. And allow the user to switch between collections.
* Authentication (OAuth)
* Connectors: User interface to configure new connections to content sources to be ingested (cloud-based storage e.g. google drive, web crawler, etc.)
* ..

## Technical: Pending improvements
* Architect for scalability (What stack would be suitable for running for 1000 concurrent users?)
* Code for production (the current prototype does not handle any errors or exceptions)
* Containerize
* Management scripts (install, backup, etc.)

# Acknowledgments
This project benefits significantly from opensource code from following entities, and the team extends its sincere appreciation to them:
* The https://ollama.ai/ project @ https://github.com/jmorganca/ollama
* [Streamlit](https://streamlit.io/), [llamaindex](https://www.llamaindex.ai/), and all libraries we are using.
* Caroline Frasca, Krista Muir and Yi Ding, for their app https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/
