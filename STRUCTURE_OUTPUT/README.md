# Structured Output Parsing with LangChain

This directory contains various scripts demonstrating how to parse and structure LLM outputs using LangChain, Google Generative AI (Gemini), and Hugging Face models. Below is a detailed guide to each file and its purpose.

## Files Guide

### 1. `ChainPydanticOutputParser.py`
Demonstrates how to use `PydanticOutputParser` inside a LangChain LCEL (LangChain Expression Language) chain with `ChatGoogleGenerativeAI`. It defines a `Student` Pydantic model and automatically parses the LLM output into this structured model using the parser at the end of the chain.

**Key Learnings & Common Pitfalls:**
- When initializing `PydanticOutputParser`, you must use the `pydantic_object` parameter (e.g., `PydanticOutputParser(pydantic_object=Student)`), not `object`.
- Because the parser is at the end of the LCEL chain, calling `chain.invoke()` directly returns the parsed Pydantic object (not an `AIMessage` or a raw string). Therefore, attempting to access `.text` on the result will cause an `AttributeError`. Instead, you can print the object directly or use `.model_dump()` to convert it to a dictionary.

### 2. `PydanticOutputParser.py`
Shows how to use `PydanticOutputParser` to parse output for a `Person` schema using `ChatGoogleGenerativeAI`. Unlike the chained version, this script manually invokes the prompt template and the LLM, then relies on the parser to format the instructions in the prompt.

### 3. `StructureOutputParser.py`
Utilizes `StructuredOutputParser` and `ResponseSchema` (from `langchain_classic.output_parsers`) to format output into a structured dictionary/JSON format for a fictional character schema using the `gemini-3.1-flash-lite` model.

### 4. `gemini_stroutputparser.py`
Demonstrates the use of `StrOutputParser` in a multi-step LCEL chain with Google Generative AI. It first generates a detailed report on a topic and then pipes the result into a second prompt to summarize the text.

### 5. `hf_output_strparser.py`
Similar to `gemini_stroutputparser.py`, but uses a Hugging Face model (`meta-llama/Meta-Llama-3-8B-Instruct`) via `ChatHuggingFace` and `HuggingFaceEndpoint` to generate and summarize text using `StrOutputParser` in a multi-step chain.

### 6. `jsonoutputparser.py`
Shows how to use `JsonOutputParser` with a Hugging Face model (`Meta-Llama-3-8B-Instruct`) to generate and parse structured JSON output representing a fictional character.

### 7. `pydantic_demo.py`
A simple demonstration of pure Pydantic functionality. It defines a `Student` model with specific field constraints (e.g., age, email, cgpa) and shows how to instantiate the model from a dictionary and export it to a JSON string.

### 8. `structure.py`
A basic Python script demonstrating the use of `TypedDict` to define a simple `Person` structure (name and age) and iterate over its items.

### 9. `with_structure_HF.py`
Attempts to use the `.with_structured_output()` method with a Hugging Face model (`ChatHuggingFace`) to extract structured review data (themes, summary, rating, sentiment, etc.) into a Pydantic `Review` model. *(Note: As commented in the script, this might not work out-of-the-box with all Hugging Face endpoints depending on structured output API support).*

### 10. `with_structure_output.py`
Demonstrates the use of `.with_structured_output()` using a `ChatGoogleGenerativeAI` model. It defines a `Review` structure using Python's `TypedDict` and `Annotated` types to directly extract structured review data from a raw text review without needing a separate parser object.

### 11. `with_structure_pydantic.py`
Similar to `with_structure_output.py`, but uses a Pydantic `BaseModel` instead of `TypedDict` for the `Review` schema. It uses `.with_structured_output(Review)` on a Google Generative AI model to extract structured data from a smartphone review and prints it as a dictionary.
