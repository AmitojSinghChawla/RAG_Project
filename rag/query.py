from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(verbose=True)  # loads GOOGLE_API_KEY

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0
)

# Prompt template
template = PromptTemplate(
    template="Answer the following question: {question}?",
    input_variables=["question"]
)


# Combine as chain
chain = template | llm

# Test query
result = chain.invoke({"question": "What is the capital of France?"})
print(result.content)
