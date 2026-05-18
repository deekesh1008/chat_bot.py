from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key="AIzaSyCZ2ApjBjLLKAWVjxJ0bVV4MH0KV82ydos",
    temperature=0.3
)

response = llm.invoke("do ?")

print(response.content)


"""User Question
      ↓
invoke()
      ↓
LangChain
      ↓
Gemini API
      ↓
Google Server
      ↓
AI generates answer
      ↓
response object
      ↓
response.content
      ↓
print()"""