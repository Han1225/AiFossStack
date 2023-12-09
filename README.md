# installation steps:
* create virtual env $python -m venv .venv
* activate it $source .venv/bin/activate
## Install required packages (Internet needed)
* $pip install --upgrade pip
* $pip install -r requirements.txt
## Your datasets 
* Data: place the knowledge files (text or pdf) that you wish to use as source inside the "data" folder. You can create subfolders. The repo contains a sample pdf feel free to replace it by your own docs.
## Your defaults
* Open the app.py file with a text editor and change the system prompt and other settings
## Start the app
$ streamlit run app.py


