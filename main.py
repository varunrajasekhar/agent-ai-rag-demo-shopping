from dotenv import load_dotenv
load_dotenv()

from agent import agent

def choose_department() -> str:
    print("Which department are you shopping for?")
    print("1. Women")
    print("2. Men")

    while True:
        choice = input("\n You:").strip().lower()
        if choice in {"1", "women"}:
            return "women"
        elif choice in {"2", "men"}:
            return "men"
        print("Please enter 1 for women or 2 for men!")

def main():
    department = choose_department()

    print(f"{department}'s department.")
    print("Try: \n"
          "Search Zara, H&M and Mango for professional, comfortable clothes under $100")
    print("\n Type 'exit' or 'quit' when you are finished")
    
    conversation = []
    while True:
        user_input = input("You:").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        grounded_input={
            "role":"user",
            "content":(f"ACTIVE_SHOPPING_DEPARTMENT: {department}\n"
                    f"SHOPPER_REQUEST: {user_input}"
                    "If this request asks for products, actually invoke search_products. Do not merely print a propsed tool call")
        }
        invoke_messages = conversation + [grounded_input]
        try:
            result = agent.invoke({"messages": invoke_messages})
            final_message = result["messages"][-1]
            content = getattr(final_message, "content", str(final_message))
            print(f"Agent: {content}")
            conversation.append({"role":"user", "content": user_input})
            conversation.append({"role":"assistant", "content": content})
        except Exception as e:
            print(f"Error invoking agent: {e}")
main()