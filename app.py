# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All

import streamlit as st

from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)


PATH = 'C:/Users/User/AppData/Local/nomic.ai/GPT4All/ggml-gpt4all-l13b-snoozy.bin'
llm = GPT4All(model=PATH, verbose=True)

embeddings = OpenAIEmbeddings()
loader = PyPDFLoader('rizz.pdf')

pages = loader.load_and_split()

store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')

vectorstore_info = VectorStoreInfo(
    name="rizz pdf",
    description="pdf contains the pickup lines and casual conversations",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('Wing AI')

prompt = st.text_input('Input your prompt here')


if prompt:
    
    response = agent_executor.run(prompt)
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 