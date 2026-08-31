from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10  # Maximum number of iterations for the agent loop
MODEL = "qwen3:1.7b"  # Specify the model to use


# ---Tools (LangChain @tool decorator)---

@tool
def get_product_price(product: str) -> float:

    """Look up the price of a product in the catalog"""
    print(f"  >> Executing get_product_price(product: '{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)  # Return 0.0 if product not found  

@tool
def apply_discount(price: float, discount_tier: str) -> float:

    """Apply a discount tier to a price and return the final price.
       Available tiers: bronze, silver, gold."""
    print(f"  >> Executing apply_discount(price: {price}, discount_tier: '{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1- discount / 100), 2) 


# --- Agent Loop ---

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools} # Construire un dictionnaire à partir d'une liste.Ça servira à Python de trouver l'outil demandé par son nom. 

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0) # We can use any model for example  openai:gpt-5 in the place of  ollama
    llm_with_tools = llm.bind_tools(tools) # C'est ici que LangChain présente au modèle la description des outils qu’il peut demander à utiliser.

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
            "You are a helpful shopping assistant."
            "You have access to a product catalog tool"
            "and a discount tool. \n\n"
            "STRICT RULES - you must follow these exactely:\n"
            "1. NEVER guess or assume any product price."
            "You MUST call get_product_price first to get the real price.\n"
            "2. Only call apply_discount aAFTER you have received"
            "a price from get_product_price. Pass the exact price" 
            "returned by get_product_price - do NOT pass a made-up number\n"
            "3. NEVER calculate discounts yourself using math."
            "Always use the apply_discount tool.\n"
            "4. If the user does not specify a discount tier, "
            "ask them which tier to use - do NOT assume one"
            )
        ),
      HumanMessage(content=question),  
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages) # Le programme Python demande à LangChain de transmettre au modèle les messages contenant les règles et la question de l’utilisateur.
                                                     # Le modèle doit retourner AIMessage
        tool_calls = ai_message.tool_calls # Le modèle demande à python l'exécution de l'outil get_product_price avec le paramètre "laptop" 

        # If no tool calls, this is the final answer (Python prend en charge la demande et vérifie si tool_calls est vide)
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content


        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0] # Python prend le premier appel 
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected] {tool_name} with args: {tool_args}  ")

        tool_to_use = tools_dict.get(tool_name) # Python cherche l'outil pour pouvoir l'exécuter
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict.")

        observation = tool_to_use.invoke(tool_args) # La variabele observation contient le résultat de l'exécution de l'outil demandé par le modèle.

        print(f"[Tool Result] {observation}")

        # Ajout du résultat à la mémoire de Python pour que le modèle puisse l'utiliser dans la prochaine itération.
        messages.append(ai_message)  # Ajout dans la mémoire la demande du modèle à Python    
        messages.append(
            ToolMessage( 
                content=str(observation),tool_call_id=tool_call_id) # Ajout dans la mémoire le résulat de l'exécution de l'outil
        )
    print("ERROR: Max iterations reached without a final answer")
    return None
       
      
if __name__ == "__main__":
   print("Hello LangChain Agent (.bind_tools)!")
   print()
   result = run_agent("What is the price of a laptop after applying a gold discount?")
