# LangChain and Agentic AI Learning Repository

This repository contains a comprehensive collection of reference implementations, exercises, and mini-applications built while learning the **LangChain** ecosystem and **Agentic AI** design patterns. It covers everything from basic model configuration to structured data extraction, semantic search, and API deployment.

---

## Workspace Structure & Detailed File Guide

The project is organized into specialized directories. Below is a detailed breakdown of every file, its purpose, and the work demonstrated within it:

### 1. [Chatmodel/](./Chatmodel)
This directory focuses on integrating chat-based language models using various commercial APIs and local pipelines.

*   **[chatmodel_gemini.py](./Chatmodel/chatmodel_gemini.py)**: Demonstrates basic integration with Google's Gemini models using the `ChatGoogleGenerativeAI` class. It loads environment variables and invokes `gemini-3.1-flash-lite` to answer a simple query.
*   **[chatmodel_hf_api.py](./Chatmodel/chatmodel_hf_api.py)**: Integrates with Hugging Face's serverless inference API (`HuggingFaceEndpoint` and `ChatHuggingFace`) to run the `meta-llama/Meta-Llama-3-8B-Instruct` model remotely, performing chat completions.
*   **[chatmodel_hf_local.py](./Chatmodel/chatmodel_hf_local.py)**: Demonstrates how to load and run a Hugging Face model (`Qwen/Qwen2.5-1.5B-Instruct`) locally using `HuggingFacePipeline` and local caching in a customized cache directory (`C:\huggingface_cache`).

### 2. [LLM/](./LLM)
This directory explores standard completion-style LLM interfaces (non-chat).

*   **[llm_gemini.py](./LLM/llm_gemini.py)**: Explores the standard text completion LLM interface using `GoogleGenerativeAI` with the `gemini-2.5-flash` model. It also provides commented reference code for interacting with OpenAI's legacy instruct models.

### 3. [PROMPTS/](./PROMPTS)
This directory is dedicated to dynamic prompt templates, prompt serialization, chat history tracking, and interactive Streamlit user interfaces.

*   **[chat.py](./PROMPTS/chat.py)**: Implements a simple terminal-based chatbot loop that maintains conversation history as a list of raw strings and continuously invokes the Gemini model.
*   **[chat_prompt_template.py](./PROMPTS/chat_prompt_template.py)**: Demonstrates the use of `ChatPromptTemplate.from_messages` to construct dynamic prompts with structured system and human roles containing placeholder variables (`{domain}`, `{topic}`).
*   **[chatbotlabelhistory.py](./PROMPTS/chatbotlabelhistory.py)**: Creates an interactive terminal chatbot that maintains context using explicit LangChain message classes (`SystemMessage`, `HumanMessage`, `AIMessage`).
*   **[langchain_promptui.py](./PROMPTS/langchain_promptui.py)**: A Streamlit web application that provides an interactive UI text box and button to query the Gemini model with customizable temperature settings.
*   **[message_data.txt](./PROMPTS/message_data.txt)**: A plain text file containing mock chat history data used for demonstration purposes.
*   **[messages.py](./PROMPTS/messages.py)**: Shows how to use `MessagesPlaceholder` to pass a dynamic list of historical messages (read from `message_data.txt`) into a `ChatPromptTemplate` along with a new user query.
*   **[prompt_template.py](./PROMPTS/prompt_template.py)**: A Streamlit-based "Research Tool" that loads a serialized prompt template from a JSON file using `load_prompt`, binds it in a chain using LangChain Expression Language (LCEL: `chain = template | model`), and generates summaries based on user UI selections.
*   **[tempelate_generator.py](./PROMPTS/tempelate_generator.py)**: Demonstrates prompt serialization by programmatically creating a `PromptTemplate` object and saving it to a local JSON file (`template.json`).
*   **[ChatPromptTemplate.py](./PROMPTS/ChatPromptTemplate.py)**: A placeholder/empty Python script.
*   **[TEMPLATE/template.json](./PROMPTS/TEMPLATE/template.json)**: The serialized JSON file containing prompt configurations, which is dynamically loaded by `prompt_template.py`.

### 4. [EMBEDDINGS/](./EMBEDDINGS)
This directory covers text embeddings generation and mathematical vector similarity operations.

*   **[embeddings_googleai.py](./EMBEDDINGS/embeddings_googleai.py)**: Introduces text embedding generation using `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2`) to convert text strings into dense float vectors with custom dimension sizes (e.g., 32).
*   **[embedding_similarity_googlegenai.py](./EMBEDDINGS/embedding_similarity_googlegenai.py)**: Implements a semantic search engine. It embeds a dataset of cricket player biography snippets and uses `scikit-learn`'s `cosine_similarity` to calculate vector distances, returning the most semantically relevant biography for a user query (e.g., "all rounder").

### 5. [STRUCTURE_OUTPUT/](./STRUCTURE_OUTPUT)
This directory focuses on schema enforcement, output parsing, and prompt chaining to guarantee structured machine-readable responses.

*   **[ChainPydanticOutputParser.py](./STRUCTURE_OUTPUT/ChainPydanticOutputParser.py)**: Demonstrates how to use `PydanticOutputParser` inside a LangChain LCEL chain with `ChatGoogleGenerativeAI`. It defines a `Student` Pydantic model and automatically parses the LLM output into this structured model using the parser at the end of the chain.
    > **⚠️ Key Learnings & Common Pitfalls:**
    > - When initializing `PydanticOutputParser`, you **must** use the `pydantic_object` parameter (e.g., `PydanticOutputParser(pydantic_object=Student)`), **not** `object`.
    > - Because the parser is at the end of the LCEL chain, `chain.invoke()` returns a parsed **Pydantic object** (not an `AIMessage`). Accessing `.text` on it will throw `AttributeError`. Use `print(result)` directly or `.model_dump()` to convert it to a dictionary.
*   **[PydanticOutputParser.py](./STRUCTURE_OUTPUT/PydanticOutputParser.py)**: Defines a Pydantic `BaseModel` schema (`Person`) and uses `PydanticOutputParser` to generate format instructions, prompt the model, and parse the output into structured Pydantic objects.
*   **[StructureOutputParser.py](./STRUCTURE_OUTPUT/StructureOutputParser.py)**: Implements structured output extraction using the legacy `ResponseSchema` and `StructuredOutputParser` classes to format and parse LLM responses.
*   **[gemini_stroutputparser.py](./STRUCTURE_OUTPUT/gemini_stroutputparser.py)**: Demonstrates sequential prompt chaining using LCEL (`template1 | model | StrOutputParser | template2 | model | StrOutputParser`), where the output of the first model is summarized by the second.
*   **[README.md](./STRUCTURE_OUTPUT/README.md)**: A comprehensive reference guide documenting every file in this directory with detailed descriptions, key learnings, and common pitfalls for each structured output approach.
*   **[hf_output_strparser.py](./STRUCTURE_OUTPUT/hf_output_strparser.py)**: Replicates the sequential chaining design pattern of `gemini_stroutputparser.py` but executes it using Hugging Face's Llama 3 model.
*   **[jsonoutputparser.py](./STRUCTURE_OUTPUT/jsonoutputparser.py)**: Uses `JsonOutputParser` without a strict Pydantic model to guide Hugging Face models to return valid, raw JSON structures representing fictional characters.
*   **[pydantic_demo.py](./STRUCTURE_OUTPUT/pydantic_demo.py)**: A standalone Python demonstration of Pydantic schemas (`BaseModel`, `Field`, and `EmailStr`) showing how data validation and JSON serialization work in pure Python.
*   **[structure.py](./STRUCTURE_OUTPUT/structure.py)**: A basic Python script demonstrating the usage and iteration of Python's built-in `TypedDict` structure.
*   **[with_structure_HF.py](./STRUCTURE_OUTPUT/with_structure_HF.py)**: Tests the compatibility and limitations of using the `with_structured_output()` method on Hugging Face API models.
*   **[with_structure_output.py](./STRUCTURE_OUTPUT/with_structure_output.py)**: Uses `with_structured_output()` with native Python `TypedDict` and `Annotated` parameters to extract structured review parameters from a long product review.
*   **[with_structure_pydantic.py](./STRUCTURE_OUTPUT/with_structure_pydantic.py)**: Extracts the same product review details but uses a robust Pydantic `BaseModel` schema with value validation constraints (e.g., rating range limits).

### 6. [CHAINS/](./CHAINS)
This directory demonstrates how to build and compose LangChain Expression Language (LCEL) chains — the core mechanism for connecting prompts, models, and parsers into a single pipeline.

*   **[simple_chain.py](./CHAINS/simple_chain.py)**: A basic LCEL chain that connects a `PromptTemplate` → `ChatGoogleGenerativeAI` → `StrOutputParser` using the pipe (`|`) operator. Also demonstrates `chain.get_graph().print_ascii()` to visualize the chain's execution graph in the terminal.
*   **[conditional_chain.py](./CHAINS/conditional_chain.py)**: Demonstrates **conditional (branching) chains** using `RunnableBranch` and `RunnableLambda`. A classifier chain first analyzes user feedback sentiment (positive/negative) using a `PydanticOutputParser` with a `Literal`-constrained schema, then routes to the appropriate response chain — a thank-you response for positive feedback or an apology response for negative feedback.
*   **[sequntial_chain.py](./CHAINS/sequntial_chain.py)**: Demonstrates a **sequential (multi-step) chain** where the output of one LLM call feeds into the next prompt. The chain is: `prompt1 | model | parser | prompt2 | model | parser`. First generates a detailed report, then extracts the top 5 crucial points from it.
*   **[parallel_chain.py](./CHAINS/parallel_chain.py)**: Demonstrates **parallel chain execution** using `RunnableParallel` to run two sub-chains concurrently — one generates short notes (via `gemini-2.5-flash`) and the other generates quiz questions (via `gemini-3.1-flash-lite`). A merge chain then combines both outputs into a single document, which is saved to a user-specified file.
*   **[streamlit_parallel_chain.py](./CHAINS/streamlit_parallel_chain.py)**: A **Streamlit web application** that wraps the parallel chain pattern in an interactive UI. Users enter a topic, and the app concurrently generates notes and MCQ quiz questions using two Gemini models, merges them, and displays the result. Also renders the chain's execution graph as a Mermaid diagram using `mermaid.ink`.

### 7. [RUNNABLES/](./RUNNABLES)
This directory explores the **Runnable** primitives (`RunnableSequence`, `RunnableParallel`) as a lower-level, explicit alternative to LCEL's pipe (`|`) operator for composing chains.

*   **[sequential_runnable.py](./RUNNABLES/sequential_runnable.py)**: Builds a sequential pipeline using the `RunnableSequence` class explicitly (instead of the `|` operator). The chain generates a joke about a given topic, then translates it into Hinglish using a second prompt. Also demonstrates `chain.get_graph().print_ascii()` to visualize the chain's execution graph.
*   **[runnable_parallel.py](./RUNNABLES/runnable_parallel.py)**: Demonstrates **multi-model parallel execution** using `RunnableParallel` with `RunnableSequence`. Runs two sub-chains concurrently — one generates a tweet using `ChatGoogleGenerativeAI` (Gemini) and the other generates a Facebook post using `ChatHuggingFace` (Llama 3 via Hugging Face API) — for the same topic.

### 8. [api/](./api)
This directory focuses on exposing LangChain models as web services.

*   **[app.py](./api/app.py)**: Implements a minimal REST API server using the FastAPI framework to prepare for exposing LangChain logic over HTTP.

### 9. Root Files
*   **[main.py](./main.py)**: A simple entrypoint script that prints the currently installed `langchain` package version to verify environment setup.
*   **[notes.md](./notes.md)**: A sample output file generated by `parallel_chain.py` — a study guide on Quantum Computing containing merged short notes and review quiz questions.

---

## Core Topics Learned & Demonstrated

1.  **LLMs & ChatModels (Commercial & Open Source)**: Setting up interfaces for both completion models (`GoogleGenerativeAI`) and chat models (`ChatGoogleGenerativeAI`), as well as local vs. remote Hugging Face pipelines.
2.  **Prompt Engineering & Serialization**: Constructing dynamic templates, managing stateful chat histories (using explicit message classes and placeholders), and serializing prompt templates to JSON for modular loading.
3.  **Text Embeddings & Semantic Search**: Converting text to mathematical vectors and computing cosine similarity using scikit-learn to perform document retrieval.
4.  **Structured JSON Extraction**: Enforcing schemas on model outputs using Pydantic, TypedDict, `StructuredOutputParser`, and the `with_structured_output` API.
5.  **Sequential Prompt Chaining**: Chaining multiple LLM calls together using LangChain Expression Language (LCEL) so that one model's output feeds the next.
6.  **LCEL Chains (Simple, Sequential, Parallel & Conditional)**: Building composable pipelines using the `|` operator to connect prompts → models → parsers, executing sub-chains concurrently with `RunnableParallel`, routing with `RunnableBranch`, and visualizing chain graphs with `get_graph().print_ascii()` and Mermaid diagrams.
7.  **Runnable Primitives**: Using explicit `RunnableSequence` and `RunnableParallel` classes for fine-grained control over chain composition, and mixing multiple model providers (Gemini + Hugging Face) in a single parallel pipeline.

---

## How to Get Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed. It is recommended to use `uv` or `pip` within a virtual environment.

### 2. Environment Setup
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### 3. Installation
Install the project dependencies:
```bash
pip install -r requirements.txt
```

### 4. Running Scripts
You can run any script directly using your Python interpreter. For example:
```bash
python Chatmodel/chatmodel_gemini.py
```

To run the Streamlit user interfaces, use the `streamlit` CLI:
```bash
streamlit run PROMPTS/langchain_promptui.py
```
