import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


#3streamlit app
st.set_page_config(page_title="Text Summarization",page_icon=":memo:")
st.title("Langchain: Summarize text from YT or Website ")
st.subheader("Enter the URL of the YouTube video or website you want to summarize")




##set groq api key and url input
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key",value="",type="password") 

llm=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key,temperature=0.7)
prompt_template="""Write a concise summary of the following content:
content:{text}"""
prompt=PromptTemplate.from_template(prompt_template)


url=st.text_input("URL",label_visibility="collapsed")

if st.button("Summarize"):
    if not validators.url(url):
        st.error("Please enter a valid URL")
    elif groq_api_key=="":
        st.error("Please enter your Groq API Key in the sidebar")
    elif not validators.url(url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Summarizing..."):
                    if "youtube" in url:
                        loader=YoutubeLoader.from_youtube_url(url,add_video_info=True)
                    else:
                        loader=UnstructuredURLLoader(urls=[url])
                    documents=loader.load()
                    
                    chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                    summary=chain.run(documents)
                    
                    st.subheader("Summary")
                    st.write(summary)
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")