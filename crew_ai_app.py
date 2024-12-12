import streamlit as st
import os
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

        SumHistory = ""

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

        Context_Filter = Agent(
            role='Context_Filter_Agent',
            goal="Filter the given context for only parts usefull to the user's question.",
            backstory="Expert in filtering and understanding user questions.",
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

            if len(st.session_state.messages) >= 2:
                # The last message should be user input, and the second to last should be the bot's response
                last_user_message = st.session_state.messages[-2]["content"]
                last_bot_message = st.session_state.messages[-1]["content"]
                SumHistory += "\n" + last_user_message + "\n" + last_bot_message
            else:
                SumHistory = ""
            
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Step 1: Embed Query Using Groq
            query_vector = ST_model.encode(user_input)

            # Step 2: Query Qdrant for Context
            results = qdrant_client.search(
                collection_name="pdf_chunks",
                query_vector=query_vector,
                limit=5
            )

            relevant_context = "\n".join(
                f"Source: {res.payload['Source']}\nText: {res.payload['text']}" for res in results)

            if not results:
                relevant_context = "No relevant context found."


            # Step 3: Define Crew Tasks
            task_define_problem = Task(
                description=f"Clarify and define the questions: {user_input}",
                expected_output="A clear and concise definition of the question.",
                agent=Question_Identifier
            )

            Task_Summarize_Session= Task(
                description=f"Summarize the session in a clear manner based on the question: \n{user_input}: \n\n{SumHistory}",
                input=task_define_problem.output,
                expected_output="A clear summarization of the session.",
                agent=BramBot
            )
            SumHistory = Task_Summarize_Session.output

            Task_Filter_Context= Task(
                description=f"filter the context:\n{relevant_context}",
                input=task_define_problem.output,
                expected_output="Clear and concise data, that is usefull and relevant to the question. Also add the source of where you found the relevant information.",
                agent=Context_Filter
            )

            task_answer_question = Task(
                description=f"Answer the user's question with usefull context in an easy to understand way you do not need to copy the context word for word if it is not needed, if no context fill in yourself. add the source of where you found the relevant information.",
                input=(task_define_problem.output, Task_Filter_Context.output, Task_Summarize_Session.output),
                expected_output="A clear answer to the full question.",
                agent=Question_Solving
            )

            task_summarize_question = Task(
                description="Summarize the full answer in a clear manner.",
                input=task_answer_question.output,
                expected_output="A clear summarization of the answer.",
                agent=BramBot
            )

            # Step 4: Create and Run Crew
            crew = Crew(
                agents=[Question_Identifier, Context_Filter, Question_Solving, BramBot],
                tasks=[task_define_problem, Task_Summarize_Session, Task_Filter_Context, task_answer_question, task_summarize_question],
                agents_config = 'config/agents.yaml',
                tasks_config = 'config/tasks.yaml', 
                verbose=True,
                memory=False,
                llm=llm
            )
            result = crew.kickoff()

            # Step 5: Display Results and Update Chat
            st.session_state.messages.append({"role": "assistant", "content": result.raw})
            with st.chat_message("assistant"):
                st.write(result.raw)

            st.write(SumHistory)

    except Exception as e:
        st.error(f"Error in Crew AI application: {e}")
