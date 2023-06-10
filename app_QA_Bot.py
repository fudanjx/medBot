from langchain.chat_models import ChatAnthropic
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import DuckDuckGoSearchRun
from streamlit_callback import StreamlitCallbackHandler
import streamlit as st
import os
import anthropic
from langchain.document_loaders import YoutubeLoader
import pandas as pd
import duckduckgo_search

class Text_Expert:
    def __init__(self):
        self.system_prompt = self.get_system_prompt()

        self.user_prompt = HumanMessagePromptTemplate.from_template("{user_question}")

        full_prompt_template = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.user_prompt]
        )

        self.chat = ChatAnthropic(model='claude-v1-100k', max_tokens_to_sample=512, streaming=True, callbacks=[StreamlitCallbackHandler()])

        self.chain = LLMChain(llm=self.chat, prompt=full_prompt_template)

    def get_system_prompt(self):
        system_prompt = """
        Answer the user question based on below context as best you can, but speaking as compasionate medical professional.
        If the information can not be found in below context, please tell: "I do not know"
        
        #####Context
        {context}
        ##### end of Context

        Begin! Remember to answer as a compansionate medical professional when giving your final answer, keep the answer brief .
        """

        return SystemMessagePromptTemplate.from_template(system_prompt)

    def run_chain(self, language, context, question):
        return self.chain.run(
            language=language, context = context, user_question=question
        )


def search_web(site, user_query):
    search = DuckDuckGoSearchRun()
    results = search.run(f"site:{site} {user_query}")
    return results


def retrieve_speciality_plugin():
    plugin_df_dase = pd.read_csv('https://www.dropbox.com/s/tqm8riwx7d8cs59/plugin_template.csv?dl=1')

    plugin_list = plugin_df_dase['plugin_site'].tolist()

    option = st.selectbox(
        '#### Select speciality plugin:',
        plugin_list)
    return option

# create a streamlit app
st.title("Medical Assistant")
st.write("(You may refresh the page to start over)")
anthropic.api_key = st.text_input("###### Enter Anthropic API Key", type="password")
os.environ['ANTHROPIC_API_KEY']= anthropic.api_key

 
# url_str = st.text_input("###### Please enter the YouTube url")



if anthropic.api_key:
        site = retrieve_speciality_plugin()
        userQuery = st.text_input("Ask a question")
        # create a button to run the model
        if st.button("Run"):
            # run the model
            tx_expert = Text_Expert()
            
            st.session_state.context = search_web(site, userQuery)
            bot_response = tx_expert.run_chain(
                'English', st.session_state.context, 
                    userQuery)

            if "bot_response" not in st.session_state:
                st.session_state.bot_response = bot_response

            else:
                st.session_state.bot_response = bot_response

    # display the response
    # if "bot_response" in st.session_state:
        # st.write(st.session_state.bot_response)
else:
    pass
