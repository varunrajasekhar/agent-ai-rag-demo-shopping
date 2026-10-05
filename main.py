from dotenv import load_dotenv

load_dotenv()

from agent import agent
import os


def choose_department() -> str:
    print("Which department are you shopping for?")
    print("1. Women")
    print("2. Men")

    while True:
        choice = input("\nYou: ").strip().lower()
        if choice in {"1", "women"}:
            return "women"
        if choice in {"2", "men"}:
            return "men"
        print("Please enter 1 for women or 2 for men!")


def main():
    department = choose_department()
    os.environ["ACTIVE_SHOPPING_DEPARTMENT"] = department
    print(f"\nShopping in the {department}'s department.")
    print(
        "Try:\n"
        "Build me a professional, comfortable complete outfit from Zara "
        "for under $200"
    )
    print("\nType 'exit' or 'quit' when you are finished.\n")

    conversation = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        grounded_input = {
            "role": "user",
            "content": (
                f"ACTIVE_SHOPPING_DEPARTMENT: {department}\n"
                f"SHOPPER_REQUEST: {user_input}\n"
                "For a complete outfit request, follow the coordinator workflow: "
                "plan the total budget, then delegate separately to top-agent, "
                "bottom-agent, and shoes-agent. Actually invoke the tools and "
                "sub-agents; do not simulate their calls."
            ),
        }

        invoke_messages = conversation + [grounded_input]

        try:
            result = agent.invoke({"messages": invoke_messages}, config={"recursion_limit":25})
            final_message = result["messages"][-1]
            content = getattr(final_message, "content", str(final_message))
            print(f"\nAgent: {content}\n")

            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": content})
        except Exception as error:
            print(f"Error invoking agent: {error}")


if __name__ == "__main__":
    main()
