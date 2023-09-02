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

####
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
        st.image(image2,use_column_width='auto')

######

st.sidebar.header('Hi, welcome!')
st.sidebar.markdown("""
The app's goal is to answer your questions regarding professional careers
and postgraduate courses offered by Tecnológico de Monterrey.
    
Ask questions to our Chatbot.
""")

####

#st.set_page_config(page_title='Tec Chatbot')
st.title('Tec Chatbot')
st.info("The majority of \"question-answering\" software operates on unstructured textual information. However, much of the world's data is actually in table form. We've developed a demo application that allows you to pose questions to data organized in tables. For this demonstration, I'm utilizing a custom dataset sourced from Tecnológico de Monterrey's official website. To get an idea of the types of questions you can ask, feel free to explore the dataset [here](https://github.com/a01110946/chatbot/blob/main/tec_de_monterrey/Corpus_de_informacion.csv). This dataset was prepared for question answering in Spanish for a Latin American audience, however, due to the ability Large Language Models have to translate language, feel free to ask your questions in English.\n\nPlease note that this application is unofficial and not affiliated with Tecnológico de Monterrey. Its sole aim is to demonstrate how Large Language Models can query tabular data.")

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
	
	#col_blank, col_text, col1, col2 = st.columns([10, 2,1,1])
	#with col_text:
		#st.text("Feedback:")
	#with col1:
		#st.button("👍", on_click=send_feedback, args=(run_id, 1))
	#with col2:
		#st.button("👎", on_click=send_feedback, args=(run_id, 0))
