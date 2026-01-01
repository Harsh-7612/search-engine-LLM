import streamlit as st 
from langchain_groq import ChatGroq
from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

#Setting up streamlit app
st.set_page_config(page_title="Text to Math solver and Data search assistant")
st.title("Text to Math Problem Solver")

groq_api_key=st.sidebar.text_input(label="Groq API Key",type='password')

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()
    
llm=ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)

##Initialize the tools:
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find the various info on the topics mentioned."
)

## Initialize the Math tool:
maths_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=maths_chain.run,
    description="A tool for answering math related questions. ONly input mathematical expressions."
)

prompt="""
You are an agent tasked for solving user's mathematical questions. Logically arrive at the solutions and provide detailed solution and display it pointwise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variable=['question'],
    template=prompt
)

# Combine all the tools:
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning",
    func=chain.run,
    description="A tool for answering logic based and reasoning questions."
)

## Initialize the agents
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool], # reasoning tool for chain of thoughts
    verbose=False,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant", "content":"Hi, I am a Math chatbot who can answer all your maths questions"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    

## lets start the interaction:
question=st.text_area("Enter your question:","Riya has 35 apples. She gives 12 apples to her friend. Later, her uncle gives her 18 more apples. How many apples does Riya have now?")

if st.button("find my answer"):
   if question:
       with st.spinner("Generate response..."):
           st.session_state.messages.append({"role":"user","content":question})
           st.chat_message("user").write(question)
           
           st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
           response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
           st.session_state.messages.append({"role":"assistant", "content":response})
           st.write("### Response:")
           st.success(response)
   else:
       st.warning("Please enter the question")       
           