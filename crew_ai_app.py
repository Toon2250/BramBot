import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM

def run_crew_ai_app(api_key, model_config):
    """
    Runs the Crew AI application after the API key and model configuration are provided.

    Parameters:
        api_key (str): The API key for authenticating with the LLM provider.
        model_config (dict): Dictionary containing model details (e.g., model ID, base URL, API key env).
    """
    try:
        # Set API key as environment variable
        os.environ[model_config["api_key_env"]] = api_key
        
        # Initialize the LLM
        llm = LLM(
            model=model_config["model"],
            base_url=model_config["base_url"],
            temperature=0.7,
        )

        # Define agents
        Question_Identifier = Agent(
            role='Question_Identifier_Agent',
            goal="Identify and refine the user's question.",
            backstory="Expert in understanding user queries.",
            verbose=False,
            allow_delegation=False,
            llm=llm,
        )

        Question_Solving = Agent(
            role='Question_Solving_Agent',
            goal="Provide a detailed answer to the user's question.",
            backstory="Expert in problem-solving.",
            verbose=False,
            allow_delegation=False,
            llm=llm,
        )

        BramBot = Agent(
            role='Summarizing_Agent',
            goal="Summarize the answer and add an interesting tidbit.",
            backstory="Helpful assistant passionate about sharing insights.",
            verbose=False,
            allow_delegation=False,
            llm=llm,
        )

        # Chat input and history
        user_input = st.chat_input("What do you want to ask the bot?")
        if "messages" not in st.session_state:
            st.session_state.messages = []  # Initialize chat history
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Define tasks
            task_define_problem = Task(
                description=f"Clarify and define the question: {user_input}",
                expected_output="A clear definition of the user's question.",
                agent=Question_Identifier,
            )
            task_answer_question = Task(
                description="Answer the clarified question.",
                expected_output="A detailed and accurate answer.",
                agent=Question_Solving,
            )
            task_summarize_question = Task(
                description="Summarize the answer.",
                expected_output="A concise and engaging summary.",
                agent=BramBot,
            )

            # Create Crew and execute
            crew = Crew(
                agents=[Question_Identifier, Question_Solving, BramBot],
                tasks=[task_define_problem, task_answer_question, task_summarize_question],
                verbose=True,
                memory=False,
                llm=llm,
            )
            result = crew.kickoff()
            
            st.session_state.messages.append({"role": "assistant", "content": result.raw})
            with st.chat_message("assistant"):
                st.write(result.raw)

    except Exception as e:
        st.error(f"Error in Crew AI application: {e}")
