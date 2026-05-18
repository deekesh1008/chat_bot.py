import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
load_dotenv()

st.set_page_config(
    page_title="AI Assistant",
    page_icon="",
    layout="centered"
)

# Gemini Model selected with API key and temperature setting
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3
)

## Custom Styling
st.markdown("""
<style>

.main {
    padding-top: 40px;
}

.stApp {
    background-color: #0f172a;
}

h1 {
    color: white;
    text-align: center;
    font-size: 42px;
}

p {
    color: #cbd5e1;
    text-align: center;
    font-size: 18px;
}

.stTextInput > div > div > input {
    border-radius: 12px;
    padding: 14px;
    border: 1px solid #22c55e;
    background-color: #111827;
    color: white;
    font-size: 16px;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 17px;
    font-weight: bold;
    background-color: #22c55e;
    color: white;
    border: none;
}

.stButton button:hover {
    background-color: #16a34a;
    color: white;
}

.chat-box {
    padding: 20px;
    border-radius: 14px;
    margin-top: 20px;
    background-color: #111827;
    color: #22c55e;
    font-size: 17px;
    line-height: 1.8;
    border: 1px solid #22c55e;
    box-shadow: 0px 0px 15px rgba(34,197,94,0.3);
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("AI Chat Assistant")

st.markdown(
    "Ask anything and get instant AI-powered responses using Gemini + LangChain."
)

# User Input
user_input = st.text_input(
    "Enter your question:",
    placeholder="Ask me anything..."
)

# Button
if st.button("Generate Response"):

    if user_input:

        with st.spinner("AI is thinking..."):

            time.sleep(1)

            response = llm.invoke(user_input)

        st.success("Response Generated Successfully")

        st.markdown("### AI Response")

        st.markdown(
            f"""
            <div class="chat-box">
            {response.content}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("Please enter a question.")