# LangChain and Agentic AI Learning Repository

This repository contains a collection of reference implementations, exercises, and mini-applications built while learning the **LangChain** ecosystem and **Agentic AI** design patterns. It covers everything from basic model configuration to structured data extraction, semantic search, and API deployment.

---

## Workspace Structure

The project is organized into the following specialized directories:

*   **`Chatmodel/`**: Integration with chat-based language models using commercial APIs and local pipelines.
*   **`LLM/`**: Basic text completion model interfaces.
*   **`PROMPTS/`**: Dynamic prompt templates, prompt serialization, chat history tracking, and Streamlit interfaces.
*   **`EMBEDDINGS/`**: Generation of vector embeddings and mathematical similarity search.
*   **`STRUCTURE_OUTPUT/`**: Schema enforcement, output parsing (Pydantic, TypedDict), and prompt-chaining pipelines.
*   **`api/`**: Hosting LangChain models as web services.

---

## Core Topics Learned

### 1. LLMs & ChatModels (Commercial & Open Source)
Explored the interface differences between text-in/text-out LLMs and message-driven ChatModels across different hosting environments:
*   **Google Generative AI**: Instantiated models like `gemini-2.5-flash` and `gemini-3.1-flash-lite` using the `langchain_google_genai` package (see [llm_gemini.py](Chatmodel/chatmodel_gemini.py) and [chatmodel_gemini.py](Chatmodel/chatmodel_gemini.py)).
*   **Hugging Face API**: Used `HuggingFaceEndpoint` to interact with remote models like Llama 3 (see [chatmodel_hf_api.py](Chatmodel/chatmodel_hf_api.py)).
*   **Local Inference**: Configured local execution pipelines using `HuggingFacePipeline.from_model_id` with models like Qwen, managing custom cache directories (`HF_HOME`) and sampling parameters (see [chatmodel_hf_local.py](Chatmodel/chatmodel_hf_local.py)).

### 2. Prompt Engineering, State & UI Integration
Developed robust prompt management strategies and built interactive interfaces to control model behavior:
*   **Prompt Templating & LCEL**: Built dynamic templates using `ChatPromptTemplate` and executed them using the LangChain Expression Language (`chain = template | model`) (see [chat_prompt_template.py](PROMPTS/chat_prompt_template.py)).
*   **Serialization**: Programmatically generated and saved prompt configurations to JSON files for better modularity (see [tempelate_generator.py](PROMPTS/tempelate_generator.py) and [prompt_template.py](PROMPTS/prompt_template.py)).
*   **Chat History Management**: Handled multi-turn conversation state using `SystemMessage`, `HumanMessage`, and `AIMessage` classes in a terminal loop (see [chatbotlabelhistory.py](PROMPTS/chatbotlabelhistory.py)), as well as `MessagesPlaceholder` for historical message lists (see [messages.py](PROMPTS/messages.py)).
*   **Streamlit UIs**: Created simple web interfaces to allow user interaction with model runs (see [langchain_promptui.py](PROMPTS/langchain_promptui.py)).

### 3. Text Embeddings & Semantic Search
Worked with vector representations of text to perform mathematical operations such as document retrieval:
*   **Vector Generation**: Used `GoogleGenerativeAIEmbeddings` to convert queries and documents into dense float vectors, experimenting with dimension reduction (see [embeddings_googleai.py](EMBEDDINGS/embeddings_googleai.py)).
*   **Cosine Similarity Matching**: Used `scikit-learn`'s `cosine_similarity` to calculate the distance between a query vector and document vectors to retrieve the most semantically relevant text from a cricket biography dataset (see [embedding_similarity_googlegenai.py](EMBEDDINGS/embedding_similarity_googlegenai.py)).

### 4. Structured JSON Extraction & Output Parsers
Learned how to guarantee that language models return structured, machine-readable JSON data instead of freeform text:
*   **Schema Enforcement**: Enforced data schemas using Pydantic's `BaseModel` and `Field` (see [with_structure_pydantic.py](STRUCTURE_OUTPUT/with_structure_pydantic.py)) as well as native Python `TypedDict` combined with `Annotated` (see [with_structure_output.py](STRUCTURE_OUTPUT/with_structure_output.py)).
*   **JSON Parsers**: Used `JsonOutputParser` along with parser format instructions to guide models to structure their outputs correctly (see [jsonoutputparser.py](STRUCTURE_OUTPUT/jsonoutputparser.py)).
*   **Sequential Chaining**: Chained prompts, models, and string parsers together so that the output of one model serves as the input of the next (see [gemini_stroutputparser.py](STRUCTURE_OUTPUT/gemini_stroutputparser.py)).

### 5. API Development & Project Infrastructure
Learned how to expose LangChain logic to exterior applications and handle development environments securely:
*   **FastAPI**: Set up a REST API framework using FastAPI (see [api/app.py](api/app.py)).
*   **Environment Management**: Used `dotenv` to load API tokens securely from a `.env` file, keeping secrets out of version control.
*   **Package Management**: Managed project dependencies using `pyproject.toml`, `requirements.txt`, and `uv.lock`.

---

## How to Get Started

### 1. Prerequisites
Ensure you have Python 3.10+ and `uv` or `pip` installed.

### 2. Environment Setup
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### 3. Installation
Install dependencies:
```bash
pip install -r requirements.txt
```
