import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OPENAI"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question: {question}")
    ]
)

# Function to generate response


def generate_response(question, api_key, engine, temperature, max_tokens):
    # Ensure API key is set
    if not api_key:
        return "Error: API key is missing!"

    # Pass API key explicitly
    llm = ChatOpenAI(
        model=engine,
        openai_api_key=api_key,  # ✅ Fix 1: Pass API key here
        temperature=temperature,  # ✅ Fix 2: Include temperature
        max_tokens=max_tokens  # ✅ Fix 3: Include max_tokens
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer


# Streamlit UI
st.title("Enhanced Q&A Chatbot With OpenAI")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# Select OpenAI model
engine = st.sidebar.selectbox("Select OpenAI model", [
                              "gpt-4o", "gpt-4-turbo", "gpt-4"])

# Adjust response parameters
temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider(
    "Max Tokens", min_value=50, max_value=300, value=150)

# Main user input
st.write("Go ahead and ask any question:")
user_input = st.text_input("You:")

# Generate response
if user_input and api_key:
    response = generate_response(
        user_input, api_key, engine, temperature, max_tokens)
    st.write(response)

elif user_input:
    st.warning("Please enter the OpenAI API Key in the sidebar.")
else:
    st.write("Please provide user input.")
