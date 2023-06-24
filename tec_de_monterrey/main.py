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
urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/tecnologico_de_monterrey-blue.jpeg', 'logo_tec_de_monterrey')
image = Image.open('logo_tec_de_monterrey')

urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/agent-image.jpg', 'agent-image')
image2 = Image.open('agent-image')
### FIN DE AGREGADO PARA IMAGENES

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#local_css("/Users/sofia/Downloads/style/style.css")

# Read the downloaded file using Pandas
#df = pd.read_excel("Corpus de información_v1.xlsx", sheet_name='Maestrías', header=0, dtype={'Maestría': str, 'Escuela': str, 'Universidad': str, 'Impartido en': list, 'Duración': str, 'Periodo': str}, engine='openpyxl')
df = pd.read_csv("Corpus_de_informacion.csv", sep=",", encoding="latin-1")

# Split the values in the column based on comma delimiter
df['Campus'] = df['Campus'].str.split(';')

# Convert the split values into a list of strings
df['Campus'] = df['Campus'].apply(lambda x: [str(value).strip() for value in x])

def tec_de_monterrey_agent_tool(input):
    pandas_agent = create_pandas_dataframe_agent(ChatOpenAI(model='gpt-3.5-turbo-16k', temperature=0), df, verbose=True)
    return pandas_agent.run(input)

python_repl = PythonREPL()

tools = [
Tool(
    name="Tecnológico de Monterrey Agent",
    func=tec_de_monterrey_agent_tool,
    description="A tool to retrieve information from Tecnológico de Monterrey. Always assume you need to use this tool to get information from the Tec. Always answer in Spanish.",
),
Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run
)
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(tools=tools, llm=ChatOpenAI(temperature=0), agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

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
        st.image(image2,use_column_width='auto')#despliega imagen
        
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