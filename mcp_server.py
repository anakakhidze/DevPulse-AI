"""GitHub tools exposed through the Model Context Protocol."""
import os

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()
mcp = FastMCP("DevPulse GitHub")


def _github_get(path: str, params: dict | None = None):
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(
        f"https://api.github.com{path}", headers=headers, params=params, timeout=20
    )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def get_github_repo_issues(repo_name: str) -> list[dict]:
    """Return up to ten open issues (excluding pull requests) for owner/repo."""
    items = _github_get(f"/repos/{repo_name}/issues", {"state": "open", "per_page": 10})
    return [
        {"number": item["number"], "title": item["title"], "state": item["state"],
         "url": item["html_url"], "created_at": item["created_at"]}
        for item in items if "pull_request" not in item
    ]


@mcp.tool()
def get_recent_commits(repo_name: str, branch: str = "main") -> list[dict]:
    """Return the five latest commits on a repository branch."""
    items = _github_get(f"/repos/{repo_name}/commits", {"sha": branch, "per_page": 5})
    return [
        {"sha": item["sha"], "author": (item.get("commit", {}).get("author") or {}).get("name"),
         "message": item.get("commit", {}).get("message"),
         "date": (item.get("commit", {}).get("author") or {}).get("date"),
         "url": item.get("html_url")}
        for item in items
    ]


@mcp.tool()
def get_merged_pull_requests(repo_name: str) -> list[dict]:
    """Return up to five recently merged pull requests."""
    items = _github_get(f"/repos/{repo_name}/pulls", {"state": "closed", "sort": "updated", "direction": "desc", "per_page": 20})
    return [
        {"number": item["number"], "title": item["title"], "merged_at": item["merged_at"],
         "user": (item.get("user") or {}).get("login"), "url": item["html_url"]}
        for item in items if item.get("merged_at")
    ][:5]


@mcp.tool()
def get_pull_request_details(repo_name: str, pr_number: int) -> dict:
    """Get details for one pull request, including its description and merge status."""
    item = _github_get(f"/repos/{repo_name}/pulls/{pr_number}")
    return {
        "number": item["number"], "title": item["title"], "body": item.get("body"),
        "state": item["state"], "merged": bool(item.get("merged_at")),
        "merged_at": item.get("merged_at"), "author": (item.get("user") or {}).get("login"),
        "base_branch": (item.get("base") or {}).get("ref"),
        "head_branch": (item.get("head") or {}).get("ref"), "url": item["html_url"],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
