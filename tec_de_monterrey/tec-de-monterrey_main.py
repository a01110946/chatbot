import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
import requests
import urllib.request
from PIL import Image

# Initialize session state for conversation history
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

# GitHub file URL
file_url = "https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/Tecnologico-de-Monterrey_Curriculum.csv"

# Send a GET request to download the file
response = requests.get(file_url)

# Save the file locally
with open("Corpus_de_informacion.csv", "wb") as file:
    file.write(response.content)

# Read CSV file and load Pandas DataFrame
df = pd.read_csv('Corpus_de_informacion.csv', encoding='ISO-8859-1')

# Initialize LLM and Pandas DataFram Agent using OpenAI Functions.
llm = ChatOpenAI(verbose=True, model="gpt-3.5-turbo-16k", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"], request_timeout=120, max_retries=2)
agent = create_pandas_dataframe_agent(llm, df, agent_type=AgentType.OPENAI_FUNCTIONS, memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True))

urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/logo-tec.png', 'logo_tec_de_monterrey')
image = Image.open('logo_tec_de_monterrey')

urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/agent-v1.png', 'agent-image')
image2 = Image.open('agent-image')

# Streamlit UI.
st.set_page_config(page_title="Tec de Monterrey - Chatbot", page_icon=":robot:", layout="wide")
with st.container():  
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image,use_column_width='auto')#despliega logo
        st.header('Tec ChatBot')
        st.markdown("Many \"question-answering\" tools focus on unstructured text, yet much data is tabular. We've created a demo app to query table-based data using a dataset from Tecnológico de Monterrey's website. Take a look at the dataset [here](https://github.com/a01110946/chatbot/blob/main/tec_de_monterrey/Tecnologico-de-Monterrey_Curriculum.csv) so you know what type of questions you can ask!\n\nThis app is unofficial and not tied to Tecnológico de Monterrey; it showcases how Large Language Models interact with tabular data.")
    with right_column:
        st.image(image2,use_column_width='auto')

st.sidebar.header('Hi, welcome!')
st.sidebar.markdown("""
The app's goal is to answer your questions regarding professional careers
and postgraduate courses offered by Tecnológico de Monterrey.
    
Ask questions to our Chatbot.
""")

#st.set_page_config(page_title='Tec Chatbot')
st.title('Tec Chatbot')
st.info("TecChat Bot can provide answers to most of your questions regarding Tecnológico de Monterrey's curriculum.")

query_text = st.text_input('Enter your question:', placeholder = 'In which campus is architecture offered?')

# Submit question and information query
result = None
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Calculating...'):
            response = agent({"input": query_text}, include_run_info=True)
            result = response["output"]
            st.session_state.past.append(query_text)
            st.session_state.generated.append(result)
            #run_id = response["__run"].run_id
if result is not None:
	st.info(result)
	
# Display past conversation
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.write(f"Bot: {st.session_state['generated'][i]}")
	st.write(f"User: {st.session_state['past'][i]}")
