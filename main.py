"""Interactive DevPulse CLI, including approval/resume for email and Slack."""
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent import graph


def _plain_text(content):
    """Render LangChain text blocks without dumping provider metadata."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        )
    return str(content)


def run_devpulse():
    user_id = input("User ID [default_user]: ").strip() or "default_user"
    config = {"configurable": {"thread_id": f"devpulse-{user_id}"}}
    print("DevPulse AI ready. Type 'exit' to quit.")
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        try:
            result = graph.invoke({"messages": [HumanMessage(content=prompt)], "user_id": user_id}, config)
            while result.get("__interrupt__"):
                request = result["__interrupt__"][0].value
                details = f"{request['target_label']}: {request['target']}\n"
                if request.get("subject"):
                    details += f"თემა: {request['subject']}\n"
                print(f"\n{request['warning']}\n{details}"
                      f"ტექსტის პრევიუ:\n{request['body_preview']}\n{request['prompt']}")
                answer = input("> ").strip()
                result = graph.invoke(Command(resume=answer), config)
        except Exception as exc:
            print(f"\nDevPulse AI: მოთხოვნა ვერ შესრულდა ({type(exc).__name__}): {exc}")
            continue
        last = result["messages"][-1]
        content = _plain_text(getattr(last, "content", None))
        if content.strip():
            print(f"\nDevPulse AI: {content}")


if __name__ == "__main__":
    run_devpulse()
