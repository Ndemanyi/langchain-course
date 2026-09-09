from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

import ollama
from langsmith import traceable


MAX_ITERATIONS = 10  # Maximum number of iterations for the agent loop
MODEL = "qwen3:1.7b"  # Specify the model to use


# ---Tools (LangChain @tool decorator)---

@traceable(run_type="tool") # ce décorateur provient de LangSmith, son rôle est de faire le monitoring afin de faire la journalisation et le débogage.
                            # Le paramètre run_type="tool" indique simplement à LangSmith de classer visuellement cette fonction comme un "outil" dans votre trace.
def get_product_price(product: str) -> float:

    """Look up the price of a product in the catalog"""
    print(f"  >> Executing get_product_price(product: '{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)  # Return 0.0 if product not found  

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:

    """Apply a discount tier to a price and return the final price.
       Available tiers: bronze, silver, gold."""
    print(f"  >> Executing apply_discount(price: {price}, discount_tier: '{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1- discount / 100), 2) 

# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#
# tools_for_llm = [get_product_price, apply_discount]
#
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#
# def get_product_price(product: str) -> float:
#     """Look up the price of a product in the catalog.
#
#     Args:
#         product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#     Returns:
#         The price of the product, or 0 if not found.
#     """

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangChain.

@traceable(name="Ollama Chat", run_type="llm") # Permet d'ajouter le traçage lors de l'appel du client Ollama.
def ollama_chat_traced(conversation_received): # Cette fonction permet à Python d’effectuer un appel à Ollama, au final c'est le modèle qui retourne la réponse
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=conversation_received) # Python appelle Ollama en lui fornissant ces 3 paramètres.


# --- Agent Loop ---

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    } 

    print(f"Question: {question}")
    print("=" * 60)

    conversation_history  = [
        {
        "role": "system",
        "content":(
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
            ),
        },
        {"role": "user", "content": question},  
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Différence 5: ollama.chat() directetly  instead of llm_with_tools.invoke() 
        response = ollama_chat_traced(conversation_received=conversation_history)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer (Python prend en charge la demande et vérifie si tool_calls est vide)
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content


        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0] # Python prend le premier appel
        # Defference 6: Attribute access ().function.name instead of dictionary access .get("name") 
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        print(f"[Tool Selected] {tool_name} with args: {tool_args}  ")

        tool_to_use = tools_dict.get(tool_name) # Python cherche l'outil pour pouvoir l'exécuter
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict.")

        # Difference 7: Direct function call instead of .invoke() method
        observation = tool_to_use(**tool_args) # Permet d'exécuter l'outil demandé par le modèle et stocke le résultat dans la variable observation.

        print(f"[Tool Result] {observation}")

        # Ajout du résultat à la mémoire de Python pour que le modèle puisse l'utiliser dans la prochaine itération.
        conversation_history.append(ai_message)  # Ajout dans la mémoire la demande du modèle à Python    
        conversation_history.append(
           {
             "role": "tool",
             "content": str(observation), 
           }
        )
    print("ERROR: Max iterations reached without a final answer")
    return None
       
      
if __name__ == "__main__":
   print("Hello LangChain Agent (.bind_tools)!")
   print()
   result = run_agent("What is the price of a laptop after applying a gold discount?")
