import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key="AIzaSyCZ2ApjBjLLKAWVjxJ0bVV4MH0KV82ydos",
    temperature=0.3
)

# Website Title
st.title("My AI English Teacher")

# User Input
user_input = st.text_input("Ask something:")

# Button
if st.button("Send"):

    if user_input:

        response = llm.invoke(user_input)

        st.write("AI Response:")
        st.write(response.content)