import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import requests
import urllib.request
from PIL import Image

# GitHub file URL
file_url = "https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/Corpus_de_informacion.csv"

# Send a GET request to download the file
response = requests.get(file_url)

# Save the file locally
with open("Corpus_de_informacion.csv", "wb") as file:
    file.write(response.content)

# Read CSV file and load Pandas DataFrame
df = pd.read_csv('Corpus_de_informacion.csv', encoding='ISO-8859-1')

# Set LLM and Pandas DataFram Agent using OpenAI Functions.
llm = ChatOpenAI(verbose=True, model="gpt-3.5-turbo-16k", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"], request_timeout=120, max_retries=2)
agent = create_pandas_dataframe_agent(llm, df, agent_type=AgentType.OPENAI_FUNCTIONS)


#from langsmith import Client
#client = Client()
#def send_feedback(run_id, score):
    #client.create_feedback(run_id, "user_score", score=score)

####
#image = Image.open('/users/sofia/downloads/tecnologico-de-monterrey-blue.png')
urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/logo-tec.png', 'logo_tec_de_monterrey')
image = Image.open('logo_tec_de_monterrey')

urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/agent-v1.png', 'agent-image')
image2 = Image.open('agent-image')
### FIN DE AGREGADO PARA IMAGENES


# SECCION DE ENCABEZADOS Y PANTALLA DE INICIO
# From here down is all the StreamLit UI.
st.set_page_config(page_title="Tec de Monterrey - Chatbot", page_icon=":robot:", layout="wide")
with st.container():  
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image,use_column_width='auto')#despliega logo
        st.header('Tec ChatBot')
        st.markdown("""
                    TecChat Bot can provide answers to most of your questions regarding
                    Tecnológico de Monterrey's curriculum.
                    """)
    with right_column:
        st.image(image2,use_column_width='auto') #despliega imagen
####

#st.set_page_config(page_title='Tec Chatbot')
st.title('Tec Chatbot')
st.info("Most 'question answering' applications run over unstructured text data. But a lot of the data in the world is tabular data! This is an attempt to create an application using [LangChain](https://github.com/langchain-ai/langchain) to let you ask questions of data in tabular format. For this demo application, we will use the Titanic Dataset. Please explore it [here](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) to get a sense for what questions you can ask. Please leave feedback on well the question is answered, and we will use that improve the application!")

query_text = st.text_input('Enter your question:', placeholder = 'In which campus is architecture offered?')
# Form input and query
result = None
with st.form('myform', clear_on_submit=True):
	submitted = st.form_submit_button('Submit')
	if submitted:
		with st.spinner('Calculating...'):
			response = agent({"input": query_text}, include_run_info=True)
			result = response["output"]
			run_id = response["__run"].run_id
if result is not None:
	st.info(result)
	col_blank, col_text, col1, col2 = st.columns([10, 2,1,1])
	with col_text:
		st.text("Feedback:")
	with col1:
		st.button("👍", on_click=send_feedback, args=(run_id, 1))
	with col2:
		st.button("👎", on_click=send_feedback, args=(run_id, 0))
