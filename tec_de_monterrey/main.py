"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.schema import BaseOutputParser
from typing import Any
from langchain.llms import Cohere
from langchain import PromptTemplate
import re
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents import ConversationalChatAgent
from langchain.agents import AgentExecutor
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

from langchain.agents import create_pandas_dataframe_agent
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import PythonREPL
import pandas as pd
import requests
import urllib.request
from langchain.chains import SimpleSequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# OPENAI_API_KEY ENVIROMENTAL VARIABLE 
import os
import openai
openai_api_key = st.secrets["OPENAI_API_KEY"]
# END ON ENVIRONMENTAL VARIABLE

#AGREGADO PARA IMAGENES
from PIL import Image
#aqui va la ruta real que pondremos en github
#sustituir esta linea cuando la imagen la subas
#image = Image.open('/users/sofia/downloads/tecnologico-de-monterrey-blue.png')
urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/tecnologico_de_monterrey-blue.jpeg', 'logo_tec_de_monterrey')
image = Image.open('logo_tec_de_monterrey')
### FIN DE AGREGADO PARA IMAGENES




# GitHub file URL
file_url = "https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/Carreras_profesionales.csv"

# Send a GET request to download the file
response = requests.get(file_url)

# Save the file locally
with open("Carreras_profesionales.csv", "wb") as file:
    file.write(response.content)

# Read the downloaded file using Pandas
#df = pd.read_excel("Corpus_de_informacion.xlsx", sheet_name='Oferta académica', header=0, dtype={'Nombre del Programa': str, 'Tipo de Programa': str, 'Escuela': str, 'Campus': list, 'Duración': str, 'Periodo': str}, engine='openpyxl')

df = pd.read_csv(filepath_or_buffer='Carreras_profesionales.csv', sep=",", header=0, encoding='latin-1')


# Split the values in the column based on comma and pipe delimiters
df['Campus'] = df['Campus'].astype(str).str.split('; ')
#df['Plan de Estudios'] = df['Plan de Estudios'].astype(str).str.split('|')

# Convert the split values into a list of strings
df['Campus'] = df['Campus'].apply(lambda x: [str(value).strip() for value in x])
#df['Plan de Estudios'] = df['Plan de Estudios'].apply(lambda x: [str(value).strip() for value in x])

#def tec_de_monterrey_agent_tool(input):
#    pandas_agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df, verbose=True)
#    return pandas_agent.run(input)

def get_answer_csv(query: str) -> str:

    chat = ChatOpenAI(temperature=0, verbose=True)

    messages = [
        SystemMessage(content="Eres un asistente virtual del Tec de Monterrey. Te gusta conversar, eres extenso en tus respuestas."),
        HumanMessage(content="Hola, quisiera solicitar información sobre la oferta académica del Tec.")
    ]
    chat(messages)

    csv_agent = create_pandas_dataframe_agent(llm=chat,
                                            df=[df],
                                            verbose=True)
    
    response = csv_agent.run(query)

    return response

def translation(response: str) -> str:

    llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])
    template = """Translate the given text from English to Spanish.

    Text to translate: {response}
    Translated text:"""
    prompt = PromptTemplate(template=template, input_variables=["text"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    translation = llm_chain.run(text=response)
    
    return translation



#--------------------------------------------------------------------------------




# SECCION DE ENCABEZADOS Y PANTALLA DE INICIO
# From here down is all the StreamLit UI.
st.set_page_config(page_title="Tec de Monterrey - Chatbot", page_icon=":robot:", layout="wide")
st.image(image) #despliega el logo
st.header('ChatBot del Tec de Monterrey')
st.subheader('Asistente que contesta tus dudas generales')
st.write("---")
######

st.sidebar.header('Hola, Bienvenido(a)')
st.sidebar.markdown("""
Esta App tiene por objeto contestar a tus dudas sobre las carreras 
profesionales así como los posgrados que tiene el Tec de Monterrey.
    
Realiza la preguntas a nuestro Chatbot.
""")
###### FIN DE PANTALLA DE INICIO



if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("Tú: ",
                               "Hola, enlista la oferta completa de carreras profesionales del Tec.", key="input")
    return input_text


user_input = get_text()

if user_input:
    data = get_answer_csv(query=user_input)
    output = translation(data)
    #agent_chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        
        