"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI

from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import PythonREPL
from langchain.agents.conversational_chat import ConversationalChatAgent
from langchain.schema import BaseOutputParser
from langchain.agents.agent import AgentExecutor
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import json
from pydantic import Any
from langchain.output_parsers import PydanticOutputParser
import re
import pandas as pd
import requests
import urllib.request

# GitHub file URL
file_url = "https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/Corpus_de_informacion.csv"

# Send a GET request to download the file
response = requests.get(file_url)

# Save the file locally
with open("Corpus_de_informacion.csv", "wb") as file:
    file.write(response.content)

#AGREGADO PARA IMAGENES
from PIL import Image
#aqui va la ruta real que pondremos en github
#sustituir esta linea cuando la imagen la subas
#image = Image.open('/users/sofia/downloads/tecnologico-de-monterrey-blue.png')
urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/logo-tec.png', 'logo_tec_de_monterrey')
image = Image.open('logo_tec_de_monterrey')

urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/agent-v1.png', 'agent-image')
image2 = Image.open('agent-image')
### FIN DE AGREGADO PARA IMAGENES


# SECCION DE ENCABEZADOS Y PANTALLA DE INICIO
# From here down is all the StreamLit UI.
#st.set_page_config(page_title="Tec de Monterrey - Chatbot", page_icon=":robot:", layout="wide")
with st.container():  
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image,use_column_width='auto')#despliega logo
        st.header('InfoChat Tec')
        st.markdown("""
                    Podemos ayudarte con todo lo que necesitas saber a cerca de los programas 
                    de estudio en el Tecnológico de Monterrey
                    """)
    with right_column:
        st.image(image2,use_column_width='auto') #despliega imagen
        
##### PRUEBA #####


SYSTEM_MESSAGE = """Asistente es un asesor que responde preguntas del Tec de Monterrey.

Cuando le preguntes algo, te responderá en base a la siguiente información disponible:

El Tec de Monterrey ofrece las siguientes carreras en Campus Guadalajara:
* ARQ-Arquitectura
* IC-Ingeniería Civil
* LED-Licenciatura en Derecho
* LRI-Licenciatura en Relaciones Internacionales
* LAD-Licenciatura en Arte Digital
* LC-Licenciatura en Comunicación
* LDI-Licenciatura en Diseño
* LTM-Licenciatura en Tecnología y Producción Musical
* IDM-Ingeniería en Ciencia de Datos y Matemáticas
* IBT-Ingeniería en Biotecnología
* IQ-Ingeniería Química
* IRS-Ingeniería en Robótica y Sistemas Digitales
* ITC-Ingeniería en Tecnologías Computacionales
* IID-Ingeniería en Innovación y Desarrollo
* IIS-Ingeniería Industrial y de Sistemas
* IM-Ingeniería Mecánica
* IMD-Ingeniería Biomédica
* IMT-Ingeniería en Mecatrónica
* LAET-Licenciatura en Estrategia y Transformación de Negocios
* LAF-Licenciatura en Finanzas
* LCPF-Licenciatura en Contaduría Pública y Finanzas
* LDE-Licenciatura en Emprendimiento
* LEM-Licenciatura en Mercadotecnia
* LIT-Licenciatura en Inteligencia de Negocios
* LBC-Licenciatura en Biociencias
* LNB-Licenciatura en Nutrición y Bienestar Integral
* LPS-Licenciatura en Psicología Clínica y de la Salud
* MC-Médico Cirujano
* MO-Médico Cirujano Odontólogo

El Tec de Monterrey ofrece las siguientes maestrías en Campus Guadalajara:
* Maestría en Gestión de la Ingeniería (MEM)
* Maestría en Ciberseguridad (MCY-M)
* Maestría en Arquitectura y Diseño Urbano (MDU-M)
* Maestría en Administración y Dirección de Empresas (tiempo parcial) (MBA)

El Tec de Monterrey ofrece los  siguientes doctorados en Campus Guadalajara:
* Doctorado en Ciencias de Ingeniería (DCI)
* Doctorado en Biotecnología (DBT)
* Doctorado en Ciencias Computacionales (DCC)
* Doctorado en Ciencias Clínicas (DCL)
"""

def make_chain():
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)

    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=ChatOpenAI(), tools=[], system_message=SYSTEM_MESSAGE, memory=memory, verbose=True)

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=[],
        memory=memory,
        verbose=True,
    )
    return agent_chain

#### TERMINA PRUEBA ####


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
    input_text = st.text_input("Tú: ", "Hola, tengo algunas preguntas sobre la oferta académica del Tec, ¿podrías ayudarme?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = agent_chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")