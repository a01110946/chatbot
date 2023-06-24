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
    description="A tool to retrieve information from Tec de Monterrey. Always assume you need to use this tool to get information from the Tec. Always answer in Spanish.",
),
Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run
)
]


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
        

##### PRUEBA #####
class NewAgentOutputParser(BaseOutputParser):
    def get_format_instructions(self) -> str:
        parser = PydanticOutputParser()
        FORMAT_INSTRUCTIONS = parser.get_format_instructions()
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Any:
        print("-" * 20)
        cleaned_output = text.strip()
        # Regex patterns to match action and action_input
        action_pattern = r'"action":\s*"([^"]*)"'
        action_input_pattern = r'"action_input":\s*"([^"]*)"'

        # Extracting first action and action_input values
        action = re.search(action_pattern, cleaned_output)
        action_input = re.search(action_input_pattern, cleaned_output)

        if action:
            action_value = action.group(1)
            print(f"First Action: {action_value}")
        else:
            print("Action not found")

        if action_input:
            action_input_value = action_input.group(1)
            print(f"First Action Input: {action_input_value}")
        else:
            print("Action Input not found")

        print("-" * 20)
        if action_value and action_input_value:
            return {"action": action_value, "action_input": action_input_value}

        # Problematic code left just in case
        if "```json" in cleaned_output:
            _, cleaned_output = cleaned_output.split("```json")
        if "```" in cleaned_output:
            cleaned_output, _ = cleaned_output.split("```")
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json"):]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[len("```"):]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[: -len("```")]
        cleaned_output = cleaned_output.strip()
        response = json.loads(cleaned_output)
        return {"action": response["action"], "action_input": response["action_input"]}
        # end of problematic code

def make_chain():
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)

    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=ChatOpenAI(), tools=[], memory=memory, verbose=True, output_parser=NewAgentOutputParser())

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
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
    output = make_chain().run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")