"""LangGraph agent with MCP-backed GitHub tools and email approval."""
import json
import os
import re
import sqlite3
import smtplib
import sys
import html
import xml.etree.ElementTree as ET
from email.message import EmailMessage
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import anyio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from state import AgentState

load_dotenv()


async def _call_github_mcp(name: str, arguments: dict):
    """Start the local MCP server and call a tool using the MCP client protocol."""
    server = Path(__file__).with_name("mcp_server.py")
    params = StdioServerParameters(command=sys.executable, args=[str(server)], env=os.environ.copy())
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            if getattr(result, "isError", getattr(result, "is_error", False)):
                raise RuntimeError("GitHub MCP tool failed: " + str(result.content))
            return "\n".join(block.text for block in result.content if getattr(block, "text", None))


def _mcp_call(name: str, **arguments):
    return anyio.run(_call_github_mcp, name, arguments)


@tool
def get_github_repo_issues(repo_name: str) -> str:
    """Get open GitHub issues for owner/repo using the GitHub MCP server."""
    return _mcp_call("get_github_repo_issues", repo_name=repo_name)


@tool
def get_recent_commits(repo_name: str, branch: str = "main") -> str:
    """Get recent commits on a repository branch using the GitHub MCP server."""
    return _mcp_call("get_recent_commits", repo_name=repo_name, branch=branch)


@tool
def get_merged_pull_requests(repo_name: str) -> str:
    """Get recently merged pull requests using the GitHub MCP server."""
    return _mcp_call("get_merged_pull_requests", repo_name=repo_name)


@tool
def get_pull_request_details(repo_name: str, pr_number: int) -> str:
    """Analyze one GitHub pull request, including its description and merge status."""
    return _mcp_call("get_pull_request_details", repo_name=repo_name, pr_number=pr_number)


@tool
def TavilySearch(query: str) -> str:
    """Search current technology news with Tavily or free Google News RSS without a key."""
    api_key = os.getenv("TAVILY_API_KEY")
    import requests
    if api_key:
        response = requests.post("https://api.tavily.com/search", json={
            "api_key": api_key, "query": query, "search_depth": "basic", "max_results": 5
        }, timeout=25)
        response.raise_for_status()
        results = response.json().get("results", [])
        return "\n".join(
            f"{item.get('title')} — {item.get('content')} ({item.get('url')})"
            for item in results
        ) or "Tavily returned no results."

    # Google News RSS is a free search feed; no API key is needed.
    params = urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    try:
        response = requests.get(
            f"https://news.google.com/rss/search?{params}",
            headers={"User-Agent": "DevPulseAI/1.0"}, timeout=25,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items = root.findall("./channel/item")[:5]
        if items:
            results = []
            for item in items:
                title = item.findtext("title", default="(untitled)")
                link = item.findtext("link", default="")
                published = item.findtext("pubDate", default="date unavailable")
                description = item.findtext("description", default="")
                description = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", description)).split())
                results.append(f"{title} — {published} — {description[:700]} ({link})")
            return "Source: Google News RSS\n" + "\n".join(results)
    except Exception as news_error:
        google_news_error = news_error

    # Keep Hacker News as a second free fallback if Google News is unavailable.
    try:
        response = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "story", "hitsPerPage": 5}, timeout=25,
        )
        response.raise_for_status()
        hits = response.json().get("hits", [])
        if hits:
            return "Source: Hacker News (Algolia search; free fallback)\n" + "\n".join(
                f"{item.get('title') or '(untitled)'} — "
                f"{item.get('url') or 'https://news.ycombinator.com/item?id=' + str(item.get('objectID'))} — "
                f"published {item.get('created_at')}; {(item.get('story_text') or '')[:700]}"
                for item in hits
            )
        return "Google News and Hacker News returned no matching stories."
    except Exception as hn_error:
        reason = locals().get("google_news_error", hn_error)
        raise RuntimeError(f"Free web search providers are unavailable: {reason}; {hn_error}") from hn_error


@tool
def SendEmail(to: str, subject: str, body: str) -> str:
    """Prepare an email digest. Sending always pauses for explicit human approval."""
    return "Email queued for approval. It will not be sent until the user approves."


@tool
def SendSlackMessage(text: str) -> str:
    """Prepare a Slack message. Sending always pauses for explicit human approval."""
    return "Slack message queued for approval. It will not be sent until the user approves."


load_dotenv()
_TOOLS = [get_github_repo_issues, get_recent_commits, get_merged_pull_requests,
          get_pull_request_details, TavilySearch, SendEmail, SendSlackMessage]
llm = None
llm_with_tools = None


def get_llm():
    """Create the free Gemini model on demand, with a useful missing-key error."""
    global llm
    if llm is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY in .env using a free key from https://aistudio.google.com/apikey"
            )
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            api_key=api_key,
        )
    return llm


def get_llm_with_tools():
    global llm_with_tools
    if llm_with_tools is None:
        llm_with_tools = get_llm().bind_tools(_TOOLS)
    return llm_with_tools
store = InMemoryStore()
PREFERENCES_DB = Path(__file__).with_name("devpulse_memory.sqlite3")

_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
_REPO_RE = re.compile(r"(?<![\w.-])([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?![\w.-])")


def _extract_preferences(text: str, previous: dict) -> dict:
    prefs = dict(previous)
    email = _EMAIL_RE.search(text)
    repo = _REPO_RE.search(text)
    if email:
        prefs["email"] = email.group(0)
    if repo:
        prefs["repo"] = repo.group(1)
    known_topics = [
        "Python", "JavaScript", "TypeScript", "React", "Next.js", "Node.js", "Java", "Go",
        "Rust", "Kotlin", "Swift", "Django", "FastAPI", "LangChain", "LangGraph", "MCP", "AI", "LLM",
    ]
    topic_text = _EMAIL_RE.sub(" ", text)
    topic_text = _REPO_RE.sub(" ", topic_text)
    found_topics = [topic for topic in known_topics
                    if re.search(r"(?<![\w])" + re.escape(topic) + r"(?![\w])", topic_text, re.I)]
    if found_topics:
        existing = prefs.get("topics", [])
        prefs["topics"] = list(dict.fromkeys([*existing, *found_topics]))
    return prefs


def load_memory_node(state: AgentState):
    user_id = state.get("user_id", "default_user")
    namespace = ("user_memory", user_id)
    item = store.get(namespace, "preferences")
    preferences = item.value if item else _read_persistent_preferences(user_id)
    latest_user_text = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
    if isinstance(latest_user_text, str):
        preferences = _extract_preferences(latest_user_text, preferences)
    return {"user_preferences": preferences, "completed_tools": [], "pending_email": None,
            "pending_slack": None, "awaiting_approval": False}


def call_model(state: AgentState):
    preferences = state.get("user_preferences", {})
    instructions = SystemMessage(content=(
        "You are DevPulse AI, a developer digest assistant. Use GitHub tools for repository facts; "
        "never invent commits, issues, or PRs. Use TavilySearch for current technology news. "
        "Use saved preferences when the request omits a repository or recipient. For requested communications, first collect and analyze "
        "the requested data, then call SendEmail or SendSlackMessage with the complete preview. Both require user approval, "
        "and neither may be called unless the user's latest message explicitly asks for that destination. "
        f"Saved user preferences: {preferences!r}. Reply in the user's language."
    ))
    response = get_llm_with_tools().invoke([instructions, *state["messages"]])
    return {"messages": [response]}


def _latest_user_request(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if message.type == "human" and isinstance(message.content, str):
            return message.content
    return ""


def _user_requested_email(state: AgentState) -> bool:
    text = _latest_user_request(state).lower()
    georgian_send = re.search(
        r"(?:ელფოსტით\s*)?(?:გამომიგზავნე|გაუგზავნე|გააგზავნე|გაგზავნე)|მეილზე\s*(?:გამომიგზავნე|გაუგზავნე|გააგზავნე)",
        text,
    )
    english_send = re.search(
        r"\b(?:send\s+(?:(?:me|it|this|the digest|the summary)\s+)*(?:an?\s+)?(?:email|e-mail)|"
        r"email\s+(?:me|it\s+to)|mail\s+me|send\b.{0,80}\bby\s+(?:email|e-mail))\b",
        text,
    )
    return bool(georgian_send or english_send)


def _user_requested_slack(state: AgentState) -> bool:
    text = _latest_user_request(state).lower()
    return bool(
        re.search(r"\b(?:send|post|share)\b.{0,50}\bslack\b|\bslack\b.{0,50}\b(?:send|post|share)\b", text)
        or re.search(r"slack.{0,30}(?:გამომიგზავნე|გაუგზავნე|გააგზავნე|გააზიარე)|"
                     r"(?:გამომიგზავნე|გაუგზავნე|გააგზავნე|გააზიარე).{0,30}slack", text)
    )


def execute_tools(state: AgentState):
    message = state["messages"][-1]
    outputs = []
    completed = list(state.get("completed_tools", []))
    pending = None
    pending_slack = None
    send_tool_count = sum(call["name"] in {"SendEmail", "SendSlackMessage"} for call in message.tool_calls)
    fetch_tools = {"get_github_repo_issues", "get_recent_commits", "get_merged_pull_requests",
                   "get_pull_request_details", "TavilySearch"}
    fetching_in_this_turn = any(call["name"] in fetch_tools for call in message.tool_calls)
    for call in message.tool_calls:
        name, args = call["name"], call["args"]
        if name == "SendEmail":
            if send_tool_count > 1:
                outputs.append(ToolMessage(
                    content="Only one communication destination can be approved per request. Ask the user whether to send by email or Slack, then prepare one message.",
                    tool_call_id=call["id"],
                ))
            elif fetching_in_this_turn:
                outputs.append(ToolMessage(
                    content="The requested data was fetched in this turn. Review those results and prepare an updated digest before asking for email approval.",
                    tool_call_id=call["id"],
                ))
            elif not _user_requested_email(state):
                outputs.append(ToolMessage(
                    content="Email not prepared: the user's latest request did not ask to send an email. Save the preference or answer the user's question without requesting email approval.",
                    tool_call_id=call["id"],
                ))
            elif not _valid_email_recipient(args.get("to")):
                outputs.append(ToolMessage(
                    content="Email not prepared: the recipient is missing or looks like a placeholder address. Ask the user for their real email address and save it before preparing the email again.",
                    tool_call_id=call["id"],
                ))
            elif not _email_recipient_was_provided(state, args.get("to")):
                outputs.append(ToolMessage(
                    content="Email not prepared: this recipient was not provided by the user or found in their saved preferences. Ask the user to specify an address before preparing the email.",
                    tool_call_id=call["id"],
                ))
            else:
                pending = {"to": args["to"], "subject": args["subject"], "body": args["body"], "tool_call_id": call["id"]}
        elif name == "SendSlackMessage":
            if send_tool_count > 1:
                outputs.append(ToolMessage(
                    content="Only one communication destination can be approved per request. Ask the user whether to send by email or Slack, then prepare one message.",
                    tool_call_id=call["id"],
                ))
            elif fetching_in_this_turn:
                outputs.append(ToolMessage(
                    content="The requested data was fetched in this turn. Review those results and prepare an updated Slack message before asking for approval.",
                    tool_call_id=call["id"],
                ))
            elif not _user_requested_slack(state):
                outputs.append(ToolMessage(
                    content="Slack message not prepared: the user's latest request did not ask to post to Slack.",
                    tool_call_id=call["id"],
                ))
            elif not os.getenv("SLACK_WEBHOOK_URL"):
                outputs.append(ToolMessage(
                    content="Slack is not configured. Set SLACK_WEBHOOK_URL in .env to an Incoming Webhook before requesting a Slack send.",
                    tool_call_id=call["id"],
                ))
            else:
                pending_slack = {
                    "channel": os.getenv("SLACK_CHANNEL_NAME", "the configured Slack channel"),
                    "text": args["text"], "tool_call_id": call["id"],
                }
        else:
            tool_fn = {"get_github_repo_issues": get_github_repo_issues,
                       "get_recent_commits": get_recent_commits,
                       "get_merged_pull_requests": get_merged_pull_requests,
                       "get_pull_request_details": get_pull_request_details,
                       "TavilySearch": TavilySearch}.get(name)
            if tool_fn is None:
                result = f"Unknown tool: {name}"
            else:
                try:
                    result = tool_fn.invoke(args)
                except Exception as exc:
                    result = f"Tool error: {exc}"
            outputs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        completed.append(name)
    return {"messages": outputs, "completed_tools": completed, "pending_email": pending,
            "pending_slack": pending_slack,
            "awaiting_approval": pending is not None or pending_slack is not None}


def human_approval_node(state: AgentState):
    email = state.get("pending_email")
    slack = state.get("pending_slack")
    if email:
        request = {
            "warning": "გაფრთხილება: აგენტი აპირებს ელ-ფოსტის გაგზავნას.",
            "target_label": "მიმღები", "target": email["to"], "subject": email["subject"],
            "body_preview": email["body"], "prompt": "გსურთ გაგრძელება? (yes/no)",
        }
        tool_call_id = email["tool_call_id"]
    else:
        request = {
            "warning": "გაფრთხილება: აგენტი აპირებს Slack შეტყობინების გაგზავნას.",
            "target_label": "არხი", "target": slack["channel"], "subject": None,
            "body_preview": slack["text"], "prompt": "გსურთ გაგრძელება? (yes/no)",
        }
        tool_call_id = slack["tool_call_id"]
    decision = interrupt(request)
    approved = str(decision).strip().lower() in {"yes", "y", "კი", "დიახ"}
    if not approved:
        denied = ("გაგზავნა უარყოფილია; ელფოსტა არ გაგზავნილა. მომხმარებელს შეუძლია შინაარსის შესწორება მოითხოვოს."
                  if email else "გაგზავნა უარყოფილია; Slack შეტყობინება არ გაგზავნილა.")
        return {"messages": [ToolMessage(content=denied, tool_call_id=tool_call_id)],
                "awaiting_approval": False, "pending_email": None, "pending_slack": None}
    try:
        if email:
            _send_email(email["to"], email["subject"], email["body"])
            outcome = "ელფოსტა წარმატებით გაიგზავნა."
        else:
            _send_slack_message(slack["text"])
            outcome = "Slack შეტყობინება წარმატებით გაიგზავნა."
    except Exception as exc:
        destination = "ელფოსტის" if email else "Slack შეტყობინების"
        outcome = f"{destination} გაგზავნა ვერ მოხერხდა: {exc}"
    return {"messages": [ToolMessage(content=outcome, tool_call_id=tool_call_id)],
            "awaiting_approval": False, "pending_email": None, "pending_slack": None}


def _send_email(to: str, subject: str, body: str):
    import ssl
    host = os.getenv("SMTP_HOST")
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM", username or "")
    if not all([host, username, password, sender]):
        raise RuntimeError("SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD and SMTP_FROM must be configured")
    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = sender, to, subject
    message.set_content(body)
    port = int(os.getenv("SMTP_PORT", "587"))
    if port == 465 or os.getenv("SMTP_USE_SSL", "").lower() in {"1", "true", "yes"}:
        with smtplib.SMTP_SSL(host, port, timeout=20, context=ssl.create_default_context()) as smtp:
            smtp.login(username, password)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(username, password)
            smtp.send_message(message)


def _send_slack_message(text: str):
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        raise RuntimeError("SLACK_WEBHOOK_URL must be configured")
    import requests
    response = requests.post(webhook, json={"text": text}, timeout=20)
    response.raise_for_status()
    if response.text.strip().lower() != "ok":
        raise RuntimeError("Slack did not confirm the message")


def _valid_email_recipient(value: str | None) -> bool:
    """Reject missing, malformed, and example/test recipients before approval."""
    if not value or not _EMAIL_RE.fullmatch(value.strip()):
        return False
    domain = value.rsplit("@", 1)[-1].lower()
    return domain not in {"example.com", "example.org", "example.net", "test.com", "localhost"}


def _email_recipient_was_provided(state: AgentState, recipient: str) -> bool:
    known = {email.lower() for email in _EMAIL_RE.findall(_latest_user_request(state))}
    saved_email = state.get("user_preferences", {}).get("email")
    if saved_email:
        known.add(saved_email.lower())
    return recipient.strip().lower() in known


def save_memory_node(state: AgentState):
    user_id = state.get("user_id", "default_user")
    namespace = ("user_memory", user_id)
    preferences = state.get("user_preferences", {})
    store.put(namespace, "preferences", preferences)
    _write_persistent_preferences(user_id, preferences)
    return {}


def _read_persistent_preferences(user_id: str) -> dict:
    with sqlite3.connect(PREFERENCES_DB) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS user_preferences (user_id TEXT PRIMARY KEY, preferences TEXT NOT NULL)"
        )
        row = connection.execute(
            "SELECT preferences FROM user_preferences WHERE user_id = ?", (user_id,)
        ).fetchone()
    return json.loads(row[0]) if row else {}


def _write_persistent_preferences(user_id: str, preferences: dict):
    with sqlite3.connect(PREFERENCES_DB) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS user_preferences (user_id TEXT PRIMARY KEY, preferences TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO user_preferences(user_id, preferences) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET preferences = excluded.preferences",
            (user_id, json.dumps(preferences, ensure_ascii=False)),
        )


def after_agent(state: AgentState) -> Literal["tools", "save_memory"]:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "save_memory"


def after_tools(state: AgentState) -> Literal["approval", "agent"]:
    return "approval" if state.get("pending_email") or state.get("pending_slack") else "agent"


def after_approval(state: AgentState) -> Literal["agent"]:
    return "agent"


workflow = StateGraph(AgentState)
workflow.add_node("load_memory", load_memory_node)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("save_memory", save_memory_node)
workflow.add_edge(START, "load_memory")
workflow.add_edge("load_memory", "agent")
workflow.add_conditional_edges("agent", after_agent, {"tools": "tools", "save_memory": "save_memory"})
workflow.add_conditional_edges("tools", after_tools, {"approval": "human_approval", "agent": "agent"})
workflow.add_conditional_edges("human_approval", after_approval, {"agent": "agent"})
workflow.add_edge("save_memory", END)
graph = workflow.compile(checkpointer=MemorySaver(), store=store)
