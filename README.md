# Chatbot Application Readme

## Please download model files from hugging face and in 'models' folder before running it. 
Link to 7B model: https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/blob/main/zephyr-7b-beta.Q4_K_M.gguf

## Please add your files [pdf, docx, csv] in 'upladed_files' folder or use file uploader


## Purpose of Each File

### 1. `rag_app.py`
- **Purpose:** This file contains a Streamlit application for a user-friendly interface to interact with the chatbot.
- **How to Run:** Start the Streamlit app by running `streamlit run app.py` in the terminal.

### 2. `data_inget.py`
- **Purpose:** This file is used to create a vector database and embeddings using the Faiss vector store. It is necessary if you dont want to use file uploader.
- **How to Run:** Run the script by executing `python data_inget.py` to generate the Faiss database.

## Running the Application

Follow these steps to set up and run the entire application:

1. **Create a Virtual Environment:**
   
   python -m venv venv
 

2. **Activate the Virtual Environment:**
   - On Windows:
    
     .\venv\Scripts\activate
     
   - On Linux/Mac:
    
     source venv/bin/activate
    

3. **Install Required Modules:**
  
   pip install -r requirements.txt
  

4. **Run Script for database:**
   - Run `python data_inget.py` to create the Faiss database.


6. **Run Streamlit App:**
   - Start the Streamlit app with `streamlit run rag_app.py` to use a graphical interface for the chatbot.

Note: Ensure that each step is executed in order for the proper functioning of the application.