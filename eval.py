"""Small LLM-as-a-judge evaluation set for DevPulse tool routing and grounding.

Run with: python eval.py
Tool selection is evaluated without executing tools, so no GitHub or email side effects occur.
"""
import json

from langchain_core.messages import SystemMessage, HumanMessage

from agent import get_llm, get_llm_with_tools

DATASET = [
    {"request": "Check recent commits and merged PRs in langchain-ai/langgraph.",
     "expected": {"get_recent_commits", "get_merged_pull_requests"}},
    {"request": "Show open issues in pallets/flask.", "expected": {"get_github_repo_issues"}},
    {"request": "Find recent news about Python 3.15.", "expected": {"TavilySearch"}},
    {"request": "Email a digest of the latest commits in langchain-ai/langgraph to dev@example.com.",
     "expected": {"get_recent_commits"}},
    {"request": "The GitHub results have already been collected: langchain-ai/langgraph commit 799c2f04 says 'keep an exit run from a checkpoint the thread moved past out of its newer branch'; merged PR #9269 has the same title. Email a brief digest of only those facts to dev@example.com.",
     "expected": {"SendEmail"}},
    {"request": "The GitHub results are ready: commit abc123 fixed checkpoint restore. Post that digest to the dev-team Slack channel.",
     "expected": {"SendSlackMessage"}},
    {"request": "Explain what the MCP protocol is.", "expected": set()},
]

FAITHFULNESS_DATASET = [
    {
        "name": "supported GitHub facts",
        "expected_grounded": True,
        "evidence": "Commit abc123 by Ana on 2026-10-10: Fix checkpoint restore. Merged PR #42: Fix checkpoint restore.",
        "digest": "Ana's abc123 commit fixed checkpoint restore. PR #42 with the same fix was merged.",
    },
    {
        "name": "fabricated GitHub facts",
        "expected_grounded": False,
        "evidence": "Commit abc123 by Ana on 2026-10-10: Fix checkpoint restore.",
        "digest": "Ana's abc123 commit fixed checkpoint restore. PR #42 was merged and added 500 stars.",
    },
    {
        "name": "empty evidence",
        "expected_grounded": False,
        "evidence": "No GitHub results were returned.",
        "digest": "Three pull requests were merged yesterday and the build passed.",
    },
]


def evaluate_tool_selection():
    rows = []
    for case in DATASET:
        try:
            response = get_llm_with_tools().invoke([
                SystemMessage(content="Select only the next tool or tools needed now. For repository/news requests without evidence, "
                                      "fetch the information first and do not select a communication tool yet. If the request already includes "
                                      "the needed evidence and explicitly asks to email it or post it to Slack, select SendEmail or SendSlackMessage. Tool calls are only being inspected; "
                                      "no tool will execute. For explanatory questions answer directly without tools."),
                HumanMessage(content=case["request"]),
            ])
            selected = {call["name"] for call in response.tool_calls}
            # All expected actions should be selected; extra tools count as an error.
            correct = selected == case["expected"]
            rows.append({"request": case["request"], "expected": sorted(case["expected"]),
                         "selected": sorted(selected), "correct": correct})
        except Exception as exc:
            rows.append({"request": case["request"], "expected": sorted(case["expected"]),
                         "error": f"LLM evaluation unavailable: {type(exc).__name__}: {exc}"})
            # A quota/auth/provider failure is shared by every remaining case.
            for pending in DATASET[len(rows):]:
                rows.append({"request": pending["request"], "expected": sorted(pending["expected"]),
                             "skipped": "Stopped after the LLM provider error above."})
            break
    evaluated = [row for row in rows if "correct" in row]
    score = sum(row["correct"] for row in evaluated) / len(evaluated) if evaluated else None
    return score, rows


def faithfulness_judge(digest: str, github_evidence: str) -> dict:
    """Ask an LLM judge to flag claims that are not supported by supplied GitHub results."""
    response = get_llm().invoke([
        SystemMessage(content="You are a strict grounding evaluator. Treat the supplied GitHub evidence as the only source of truth. "
                              "A claim is unsupported if the evidence does not explicitly support it. "
                              "Return only a JSON object with keys score (number from 0 to 1; 1 means fully supported), "
                              "unsupported_claims (array of strings), and rationale (string)."),
        HumanMessage(content=f"GITHUB EVIDENCE:\n{github_evidence}\n\nDIGEST:\n{digest}"),
    ])
    try:
        content = response.content
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        if isinstance(content, str):
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
        result = json.loads(content)
        score = max(0.0, min(1.0, float(result["score"])))
        return {"score": score, "unsupported_claims": result.get("unsupported_claims", []),
                "rationale": result.get("rationale", "")}
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return {"score": 0, "unsupported_claims": [], "rationale": str(response.content)}


def evaluate_faithfulness():
    rows = []
    for case in FAITHFULNESS_DATASET:
        result = faithfulness_judge(case["digest"], case["evidence"])
        classified_grounded = result["score"] >= 0.8 and not result["unsupported_claims"]
        rows.append({"name": case["name"], "expected_grounded": case["expected_grounded"],
                     "classified_grounded": classified_grounded,
                     "correct": classified_grounded == case["expected_grounded"], **result})
    accuracy = sum(row["correct"] for row in rows) / len(rows) if rows else None
    return accuracy, rows


if __name__ == "__main__":
    accuracy, details = evaluate_tool_selection()
    if accuracy is None:
        print("Tool Selection Accuracy: unavailable (no LLM cases completed)")
    else:
        successful = sum("correct" in row for row in details)
        print(f"Tool Selection Accuracy: {accuracy:.0%} ({successful}/{len(DATASET)} cases evaluated)")
    print(json.dumps(details, indent=2, ensure_ascii=False))
    try:
        faithfulness_accuracy, faithfulness_rows = evaluate_faithfulness()
        if faithfulness_accuracy is None:
            print("Faithfulness Judge Classification: unavailable (no cases evaluated)")
        else:
            correct = sum(row["correct"] for row in faithfulness_rows)
            print(f"Faithfulness Judge Classification: {faithfulness_accuracy:.0%} ({correct}/{len(faithfulness_rows)} cases)")
        print(json.dumps(faithfulness_rows, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"Faithfulness evaluation unavailable: {type(exc).__name__}: {exc}")
