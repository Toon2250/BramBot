import streamlit as st
import os
import openai
from qdrant_client import QdrantClient
from crewai import Agent, Task, Crew, LLM
from sentence_transformers import SentenceTransformer

def run_crew_ai_app(api_key, model_config, qdrant_key, qdrant_url):
    """
    Runs the Crew AI application integrated with Groq and Qdrant.

    Parameters:
        api_key (str): Groq API key for model access.
        qdrant_key (str): Qdrant API key.
        qdrant_url (str): URL for Qdrant service.
        openai_key (str): OpenAI API key (if needed).
    """
    try:
        # Set up API keys
        os.environ[model_config["api_key_env"]] = api_key

        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

        ST_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        llm = LLM(
            model=model_config["model"],
            base_url=model_config["base_url"],
            temperature=0.7,
        )

        # Define agents with Groq LLM
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
            goal="""Summarize the solved question in a clear way.""",
            backstory="""You are a helpful assistant passionate about AI.""",
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

            # Step 1: Embed Query Using Groq
            query_vector = ST_model.encode(user_input)

            # Step 2: Query Qdrant for Context
            results = qdrant_client.search(
                collection_name="pdf_chunks",
                query_vector=query_vector,
                limit=3
            )
            relevant_context = "\n".join(res.payload["text"] for res in results)

            if not results:
                relevant_context = "No relevant context found."

            # Step 3: Define Crew Tasks
            task_define_problem = Task(
                description=f"Clarify and define the questions: {user_input}\n\nContext:\n{relevant_context}",
                expected_output="A clear and concise definition of the question.",
                agent=Question_Identifier
            )

            task_answer_question = Task(
                description=f"Answer the user's question with full context:\n\n{relevant_context}",
                expected_output="A clear answer to the full question.",
                agent=Question_Solving
            )

            task_summarize_question = Task(
                description="Summarize the full answer in a clear manner.",
                expected_output="A clear summarization of the answer.",
                agent=BramBot
            )

            # Step 4: Create and Run Crew
            crew = Crew(
                agents=[Question_Identifier, Question_Solving, BramBot],
                tasks=[task_define_problem, task_answer_question, task_summarize_question],
                verbose=True,
                memory=False,
                llm=llm
            )
            result = crew.kickoff()

            # Step 5: Display Results and Update Chat
            st.session_state.messages.append({"role": "assistant", "content": result.raw})
            with st.chat_message("assistant"):
                st.write(result.raw)

    except Exception as e:
        st.error(f"Error in Crew AI application: {e}")
