from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict , Annotated


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# schema for structured output

class Review(TypedDict):
    summary: str
    sentiment: str
    name: Annotated[str, "this is the name of the reviewer"]
    length: Annotated[int, "this is the length of the review in number of words"]
    cons: Annotated[str, "this is the cons of the product mentioned in the review"]
    pros: Annotated[str, "this is the pros of the product mentioned in the review"]
    time: Annotated[float, "this is the time take by the model to generate the output in minutes"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("Product: Apple iPhone 15 Reviewer: Rahul Sharma Review:I have been using the iPhone 15 for almost two months, and my experience has been very good so far. The camera quality is excellent, especially in daylight photography. Battery backup easily lasts a full day with normal usage. The phone feels premium and lightweight in hand.The performance is very smooth while gaming, multitasking, and using social media apps. I also noticed that the device heats less compared to my previous phone. The display quality is sharp, and watching videos feels amazing.One thing I did not like much is the charging speed because it still feels slower compared to some Android phones in the same price range. Apart from that, everything else is impressive.Overall, I would definitely recommend the iPhone 15 to users who want a premium smartphone with strong camera performance and smooth user experience")

print(type(result))
'''
print(result['summary'])
print(result['sentiment'])'''
print(result['name'])
print(result['cons'])
print(result['pros'])
print(result['time'])