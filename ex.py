import streamlit as st
import os
from qdrant_client import QdrantClient
from crewai import Agent, Task, Crew, LLM
from sentence_transformers import SentenceTransformer

def run_crew_ai_app(api_key, model_config, qdrant_key, qdrant_url, use_docs):
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

        if use_docs:
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

        ST_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        MessageList = st.session_state.messages

        llm = LLM(
            model=model_config["model"],
            base_url=model_config["base_url"],
            temperature=0.5,
        )

        # Define agents with Groq LLM
        Question_Identifier = Agent(
            role='Question_Identifier_Agent',
            goal="Identify and refine the user's question.",
            backstory="A friendly and curious expert who loves unraveling what users really mean.",
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
            goal="Summarize the solved question in a conversational, user-friendly manner.",
            backstory="A cheerful assistant who enjoys explaining things clearly and helping others learn.",
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
            if use_docs:
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
                description=f"Clarify and define the questions: {user_input}",
                expected_output="A clear and conversational understanding of what the user is asking, rephrased in a way that's easy to follow.",
                agent=Question_Identifier
            )

            Task_Summarize_Session= Task(
                description=f"Summarize the session in a clear manner based on the question: \n{user_input} \n\n{MessageList}",
                input=task_define_problem.output,
                expected_output="A friendly recap of the discussion so far, highlighting the key points and what has been addressed.",
                agent=BramBot
            )

            if use_docs:
                Task_Filter_Context = Task(
                    description=f"Filter the context:\n{relevant_context}",
                    input=task_define_problem.output,
                    expected_output="A refined selection of the most relevant and useful information to help answer the user's question effectively.",
                    agent=Context_Filter
                )
                task_answer_context_question = Task(
                    description=f"Answer the user's question with full context, if no context fill in yourself.",
                    input=(task_define_problem.output, Task_Filter_Context.output, Task_Summarize_Session.output),
                    expected_output="A thoughtful, detailed, and easy-to-understand answer that directly addresses the user's question, incorporating any available context.",
                    agent=Question_Solving
                )
            else:
                task_answer_question = Task(
                    description=(
                            f"Answer the user's question in the following structured format: "
                            f"\n\n1. **Brief Overview**: Provide a concise definition or summary. "
                            f"\n2. **Key Characteristics**: List important details as bullet points, focusing on traits or facts relevant to the query."
                        ),
                    input=task_define_problem.output,
                    expected_output="A concise and accurate answer to the user's query, unless the query requires detailed explanation.",
                    agent=Question_Solving
                )


            if use_docs:
                task_summarize_question = Task(
                    description="Summarize the full answer in a clear manner.",
                    input=task_answer_context_question.output,
                    expected_output="A concise, conversational summary of the answer that makes it easy for the user to understand the key points.",
                    agent=BramBot
                )
            else:
                task_summarize_question = Task(
                    description="Summarize the full answer in a clear manner.",
                    input=task_answer_question.output,
                    expected_output="A concise, conversational summary of the answer that makes it easy for the user to understand the key points.",
                    agent=BramBot
                )

            # Step 4: Create and Run Crew
            if use_docs:
                agents = [Question_Identifier, Context_Filter, Question_Solving, BramBot]
                tasks = [
                    task_define_problem,
                    Task_Summarize_Session,
                    Task_Filter_Context,
                    task_answer_context_question,
                    task_summarize_question,
                ]
            else:
                agents = [Question_Identifier, Question_Solving, BramBot]
                tasks = [
                    task_define_problem,
                    Task_Summarize_Session,
                    task_answer_question,
                    task_summarize_question,
                ]
            crew = Crew(
                agents=agents,
                tasks=tasks,
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
