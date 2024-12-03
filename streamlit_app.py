import streamlit as st
from groq import Groq
import os
import traceback
from crewai import Agent, Task, Crew, LLM

if "api_key" not in st.session_state:
    st.session_state.api_key = None  # Initialize API key in session state

try:
    llm = LLM(
    model="llama3-70b-8192",
    temperature=0.7,
    base_url="https://api.groq.com/openai/v1",
    api_key=st.session_state.api_key
)
    
except Exception as e:
    st.error(f"Error initializing ChatGroq: {e}")

Question_Identiefier = Agent(
        role='Question_Identiefier_Agent',
        goal="""You identify what the question is and you add other parts which is needed to answer the question.""",
        backstory="""You are an expert in understanding and defining questions. 
            Your goal is to extract a clear, concise questio, statements from the user's input.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


Question_Solving = Agent(
        role='Question_Solving_Agent',
        goal="""You solve the questions.""",
        backstory="""You are an expert in understanding and solving questions. 
            Your goal is to answer the questions in a clear, concise statement from the input, 
            ensuring the answers is clear.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )



BramBot = Agent(
        role='Summerazing_Agent',
        goal="""Summerize the solved question, in a clear way. If the question is about AI, DL, NLP you add ab interesting tidbit about the topic""",
        backstory="""You are a helpful assistant called BramBot, and you are very passionate about Artificial intelligence,
        Deep Learning and Natural language processing. You also like coffee, never call it a cup of joe.
        You also like giving some interesting tidbits about AI, DL, NLP wich have something to do with the topic.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


# Title and app description
st.title("💬 BramBot")
st.write(
    "This is a chatbot that uses Groq's powerful models to generate responses. "
    "To use this app, simply enter your message below and get an instant response from Groq's AI."
)
st.session_state.api_key = st.text_input(
    "Enter your Groq API Key:",  # Input prompt
    value=st.session_state.api_key or "",  # Pre-fill if previously entered
    type="password",  # Hide input for security
    placeholder="Your API Key here"  # Placeholder for guidance
)

# Step 5: Chatbot functionality
# Initialize the chat history if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores user and bot messages

# Display previous chat messages in the app
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # Display messages as user or assistant
        st.markdown(message["content"])

# Input field for user messages
user_input = st.chat_input("What do you want to ask the bot?")  # Input box for user queries

if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    task_define_problem = Task(
        description="""Clarify and define the questions, 
            including identifying the problem type and specific requirements.
            
            Here is the user's problem:

            {ml_problem}
            """.format(ml_problem=user_input),
        agent=Question_Identiefier,
        expected_output="A clear and concise definition of the questions.")
    
    task_answer_question= Task(
        description="""Answer and fully clarify the user's question.""",
        agent=Question_Solving,
        expected_output="A clear answer of the full question."
        )
    
    task_summerize_question = Task(
        description="""Summerize the full answer in clear manner.""",
        agent=BramBot,
        expected_output="A clear summerization of the answer."
        )
    
    crew = Crew(
            agents=[Question_Identiefier, Question_Solving, BramBot], #, Summarization_Agent],
            tasks=[task_define_problem, task_answer_question, task_summerize_question], #, task_summarize],
            verbose=True
        )

    result = crew.kickoff()

    with st.chat_message("assistant"):
        st.write(result.raw)