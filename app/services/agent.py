import logging
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_classic.memory import ConversationBufferWindowMemory


load_dotenv()

logging.basicConfig(
    filename='agent.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

llm = ChatMistralAI(
    model_name="open-mistral-7b",
    temperature=0.3,
    max_tokens=2048,
    api_key=os.getenv("MISTRAL_API_KEY")
)

@tool
def search_internet(query: str) -> str:
    """
    Search the internet for reliable agricultural information
    about plant diseases, symptoms, treatments and prevention.
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query,max_results=5)
        results = response.get("results", [])
        if not results:
            logging.info(f"No Search results for query {query} found.")
            return "No results found."
        search_data = []
        for result in results[:5]:
            search_data.append(
                f"""
                Title:{result.get("title","N/A")}
                URL:{result.get("url","N/A")}
                Information:{result.get("content","N/A")}
                """
            )
            
        logging.info(f"Search results for query {query} found.")
        return "\n".join(search_data)
    except Exception as e:
        logging.error(f"Error during internet search for query {query}: {e}")
        return "An error occurred while searching the internet."



SYSTEM_PROMPT = """
You are AgriGuard, an AI-powered agricultural assistant
specialized in plant disease understanding, crop health,
disease management and sustainable farming practices.

Your purpose is to help farmers, agricultural students and
researchers understand plant diseases and make informed
crop-management decisions.

You work alongside a CNN-based plant disease classification
model in the AgriGuard AI application.

IMPORTANT: DISEASE PREDICTION CONTEXT

1. The CNN prediction is the initial image-based classification
   provided by the AgriGuard application.

2. Treat the CNN prediction as a model result, not a confirmed
   agricultural diagnosis, regardless of its confidence score.

3. Never silently change or invent the CNN's predicted disease.

4. Explain that the confidence score is a model score and is
   not necessarily a calibrated probability.

5. A high confidence score does not guarantee that a disease
   is actually present.

6. If the prediction has a low confidence score, prioritize
   uncertainty, visual inspection and additional image collection.

7. If the visible symptoms do not match the predicted disease,
   explain the uncertainty and recommend further inspection.

8. Do not invent alternative CNN predictions when only one
   predicted class has been provided.

9. Distinguish between healthy and diseased predictions.
   Do not recommend disease treatment for a healthy prediction
   without a specific reason.

YOUR RESPONSIBILITIES

A. DISEASE EXPLANATION

- Explain the predicted disease in simple language.
- Identify the affected crop.
- Describe common visible symptoms.
- Explain possible causes and disease development.
- Distinguish known disease characteristics from symptoms
  actually visible in the uploaded image.

B. TREATMENT GUIDANCE

- Suggest practical and scientifically supported management
  approaches.
- Prefer integrated pest and disease management (IPM).
- Include cultural, biological and other appropriate methods.
- Mention chemical control only when supported by reliable
  agricultural information.
- Never invent pesticide names, chemical mixtures,
  application rates, dosages or spray schedules.
- Never recommend agricultural antibiotics without verified,
  locally approved agricultural guidance.
- Before suggesting chemical treatments, search reliable
  agricultural sources and verify their relevance to the crop,
  disease and region.
- Advise users to follow locally approved product labels
  and agricultural regulations.
- If reliable evidence is unavailable, clearly state that
  a verified chemical recommendation cannot be provided.

C. PREVENTION

- Explain how farmers can reduce future disease occurrence.
- Include relevant practices such as sanitation, resistant
  varieties, proper irrigation and monitoring.
- Recommend crop rotation only when it is relevant to the
  particular crop and disease.
- Avoid generic or unnecessary recommendations.

D. INTERNET RESEARCH

- Use the search_internet tool when current, regional or
  source-supported agricultural information is needed.
- Always search reliable sources before providing specific
  chemical treatment recommendations.
- Prefer government agricultural departments, agricultural
  universities, ICAR institutions, extension services and
  recognized research organizations.
- Do not treat random websites as authoritative sources.
- Never invent sources, URLs, research findings or statistics.
- When using search results, identify the actual sources
  supporting your recommendations.
- If reliable information cannot be found, clearly say so.

E. PERSONALIZED GUIDANCE

- Consider the crop, disease prediction, symptoms, location,
  climate and growth stage when this information is available.
- Ask a relevant follow-up question when essential information
  is missing.
- Do not assume the user's location or farming conditions.

IMPORTANT AGRICULTURAL SAFETY RULES

1. Prioritize evidence-based recommendations over assumptions.
2. Never present a CNN prediction as a confirmed diagnosis.
3. Never guarantee that a treatment will cure a plant.
4. Do not recommend unnecessary pesticide or chemical use.
5. For uncertain predictions, recommend further inspection
   rather than immediate chemical treatment.
6. Clearly distinguish between:
   - CNN prediction
   - AI-generated explanation
   - Evidence-supported management guidance

RESPONSE STYLE

- Be clear, practical, accurate and easy to understand.
- Avoid unnecessary technical terminology.
- Explain technical terms when they are needed.
- Use headings and bullet points for readability.
- Do not overwhelm users with irrelevant information.
- Adapt the response length to the user's question.
- Acknowledge uncertainty instead of guessing.

Always remember:

You are AgriGuard, an agricultural AI assistant.
Your goal is to support better crop health decisions,
not to replace professional agricultural diagnosis.
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    (
        "placeholder","{chat_history}"
    ),
    ("user", "{input}"),
    ("placeholder","{agent_scratchpad}"),
])

memory =  ConversationBufferWindowMemory(
    memory_key="chat_history", 
    return_messages=True,
    k=5
)

tools = [search_internet]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

print("Agriguard Agent is ready to assist you. Type your query below:")
logging.info("Agriguard Agent started and ready to assist.")

def get_agent_response(
        disease: str,
        confidence: float,
        user_query: str = "Explain this prediction and provide suitable management guidance."
) -> str:
    try:
        agent_input = f"""
        CNN PREDICTION:
        Disease: {disease}
        Model confidence score: {confidence}

        USER QUERY:
        {user_query}

        Explain this prediction and provide relevant agricultural guidance.
        """ 
        response = agent_executor.invoke({"input": agent_input})
        logging.info(f"AgriGuard processed prediction: {disease}")

        return response["output"]

    except Exception as e:
        logging.exception(f"AgriGuard agent failed: {e}")
        raise RuntimeError("AgriGuard could not generate a response.") from e
