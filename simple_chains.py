from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="generate 5 facts about {topic}",
    input_variables=["topic"]

)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3
)

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"topic": "space"})
print(response)

chain.get_graph().print_ascii() # with the help of that we can see the flow of the chain and how the data is passed from one component to another.  

