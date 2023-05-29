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

# Import Excel file, specify sheet name and range of columns to import
df = pd.read_excel('Corpus de información.xlsx', sheet_name='Maestrías', header=0, dtype={'Maestría': str, 'Escuela': str, 'Universidad': str, 'Impartido en': list, 'Duración': str, 'Periodo': str})
#agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df, verbose=True)

def tec_de_monterrey_agent_tool(input):
    return tec_de_monterrey_agent_tool.run(input)

python_repl = PythonREPL()

tools = [
Tool(
    name="Tecnológcio de Monterrey Agent",
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

'''
def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], temperature=0)
    #agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df, verbose=True)
    #agent = initialize_agent(tools, llm, agent=AgentType.conversational-react-description, verbose=True)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()
'''

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