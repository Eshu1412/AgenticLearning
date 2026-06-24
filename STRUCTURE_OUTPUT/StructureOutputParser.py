from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1. Define the schema for the structured output
response_schemas = [
    ResponseSchema(name="name", description="The name of the fictional character"),
    ResponseSchema(name="profession", description="The profession of the character"),
    ResponseSchema(name="city", description="The city where the character lives"),
    ResponseSchema(name="age", description="The age of the character", type="int")
]

# 2. Initialize the structured output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# 3. Create the prompt template
prompt = PromptTemplate(
    template="Give me fictional character details.\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": format_instructions}
)

# 4. Initialize model
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

# 5. Build and invoke the chain
chain = prompt | model | output_parser
result = chain.invoke({})

print(result)
