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

# GitHub file URL
file_url = "https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/Carreras_profesionales.csv"

# Send a GET request to download the file
response = requests.get(file_url)

# Save the file locally
with open("Carreras_profesionales.csv", "wb") as file:
    file.write(response.content)


# Read the downloaded file using Pandas
#df = pd.read_excel("Corpus de información_v1.xlsx", sheet_name='Maestrías', header=0, dtype={'Maestría': str, 'Escuela': str, 'Universidad': str, 'Impartido en': list, 'Duración': str, 'Periodo': str}, engine='openpyxl')
df = pd.read_csv("Carreras_profesionales.csv", sep=",", encoding="latin-1")

# Split the values in the column based on comma delimiter
df['Campus'] = df['Campus'].str.split('; ')

# Convert the split values into a list of strings
df['Campus'] = df['Campus'].apply(lambda x: [str(value).strip() for value in x])

def tec_de_monterrey_agent_tool(input):
    pandas_agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df, verbose=True)
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

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Tec de Monterrey - Chatbot", page_icon=":robot:")
st.header("Tec de Monterrey - Chatbot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("Tú: ", "Hola, ¿cómo estás?", key="input")
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