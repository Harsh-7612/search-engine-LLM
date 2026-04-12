### Text to Math Problem Solver

Summary:  
Built an LLM-powered intelligent problem-solving assistant using Groq-hosted Llama 3.3 70B integrated with LangChain agents, combining Wikipedia retrieval, mathematical computation, and custom reasoning chains to dynamically route user queries through specialized tools for context-aware answer generation. Developed an interactive Streamlit interface with persistent chat history and tool-based agent orchestration for real-time multi-step reasoning and information retrieval.

----------------------------------------------------------------------------------------------------------------------
Detailed:  
The app is a Streamlit-based “Text to Math Problem Solver” that uses Groq-powered Llama, Wikipedia lookup, calculator reasoning, and a custom reasoning chain inside one agent.

I built a Streamlit app and set it up as a text-to-math problem solver and data search assistant. I gave the app a title, created a sidebar input for the Groq API key, and blocked execution until the key was provided.
I connected the app to Groq using ChatGroq with the model llama-3.3-70b-versatile. This became the core LLM powering the assistant.
I added a Wikipedia tool using WikipediaAPIWrapper so the assistant could search for supporting information when needed. I registered it as a LangChain Tool with a clear description for internet-style lookup.
I added a calculator tool using LLMMathChain so the agent could solve mathematical expressions and arithmetic questions more reliably.
I created a separate reasoning tool using an LLMChain with a custom prompt that asks the model to solve the user’s question logically and present the answer pointwise.
I combined the Wikipedia tool, calculator tool, and reasoning tool into a single zero-shot ReAct agent using initialize_agent(...). I also enabled handle_parsing_errors=True so the app could recover better when the model response format was imperfect.
I used Streamlit session state to keep the conversation history alive across turns. I initialized the chat with a default assistant message so the interface already felt like a chatbot when it opened.
I built the main user interaction around a text area where the user can enter a question and a button labeled “find my answer” to trigger the response. I also included a sample arithmetic question by default to demonstrate how the app works.
When the user clicks the button, I append the message to the chat history, display it in the UI, and run the agent with a Streamlit callback handler so the response generation feels interactive.
After the agent returns an answer, I store it back into session state and show the final result with a success message. If no question is entered, I show a warning asking the user to type one first.

In simple terms, I built an LLM-powered Streamlit assistant that can reason over math questions, use Wikipedia for retrieval, and route tasks through the right tool automatically.
