import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM

def run_crew_ai_app(api_key):
    """
    Runs the Crew AI application after the API key is provided.

    Parameters:
        api_key (str): The API key for authenticating with Groq's LLM.
    """
    try:
        os.environ["GROQ_API_KEY"] = api_key
        groq_api_key = os.environ.get('GROQ_API_KEY')

        # Initialize the LLM
        llm = LLM(
            model="groq/llama3-70b-8192",
            temperature=0.7,
            base_url="https://api.groq.com/openai/v1",
        )

        # Define agents and tasks
        Question_Identifier = Agent(
            role='Question_Identifier_Agent',
            goal="""You identify what the question is and add other parts needed to answer it.""",
            backstory="""You are an expert in understanding and defining questions.""",
            verbose=False,
            allow_delegation=False,
            llm=llm,
        )

        Question_Solving = Agent(
            role='Question_Solving_Agent',
            goal="""You solve the questions.""",
            backstory="""You are an expert in solving questions.""",
            verbose=False,
            allow_delegation=False,
            llm=llm,
        )

        BramBot = Agent(
            role='Summarizing_Agent',
            goal="""Summarize the solved question in a clear way and add an interesting tidbit.""",
            backstory="""You are a helpful assistant passionate about AI.""",
            verbose=False,
            allow_delegation=False,
            llm=llm,
        )

        manager = Agent(
            role="Project Manager",
            goal="Efficiently manage the crew and ensure high-quality task completion",
            backstory="You're an experienced project manager.",
            llm=llm,
            allow_delegation=True,
        )
   
        # Input from user
        user_input = st.text_input("What do you want to ask the bot?")

        if user_input:

            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)


            task_define_problem = Task(
                    description=f"Clarify and define the questions: {user_input}",
                    expected_output="A clear and concise definition of the question.",
                    agent=Question_Identifier
                )

            task_answer_question = Task(
                    description="Answer and fully clarify the user's question.",
                    expected_output="A clear answer to the full question.",
                    agent=Question_Solving

                )

            task_summarize_question = Task(
                    description="Summarize the full answer in a clear manner.",
                    expected_output="A clear summarization of the answer.",
                    agent=BramBot
                )     

            crew = Crew(
                agents=[Question_Identifier, Question_Solving, BramBot],
                tasks=[task_define_problem, task_answer_question, task_summarize_question],
                verbose=True,
                memory=False,
                llm=llm,

            )

            result = crew.kickoff()
            with st.chat_message("assistant"):
                st.write("Result:", result.raw)

    except Exception as e:
        st.error(f"Error in Crew AI application: {e}")
