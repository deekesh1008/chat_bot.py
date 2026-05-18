from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate notes for the topic {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a quiz (5 questions) for the topic:\n{topic}",
    input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="Merge the following notes and quiz:\n\nNotes:\n{topic}\n\nQuiz:\n{quiz}",
    input_variables=["topic", "quiz"]
)

parallel_chain = RunnableParallel(
    topic=prompt1 | model | parser,
    quiz=prompt2 | model | parser
)

merge_chain = prompt3 | model | parser

pa = parallel_chain | merge_chain

response = pa.invoke({
    "topic": "linear regression"
})

print(response)