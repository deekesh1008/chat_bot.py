from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on the topic {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Extract only 2 important points from the following report:\n{text}",
    input_variables=["text"]
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

response = chain.invoke({
    "topic": "Unemployment in U.S.A"
})

print(response)

