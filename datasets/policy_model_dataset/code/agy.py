import asyncio
import json
import re
from typing import Literal


async def agy_call(
    system_prompt: str,
    user_prompt: str = "",
    model: str = "gemini-3.8-flash-medium",
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    json_schema: dict | None = None,
) -> dict | list:
    """
    Call the local agy CLI non-interactively to generate structured JSON output.

    Args:
        system_prompt: System prompt containing role and schema instructions.
        user_prompt: User instruction or input payload data.
        model: Model name for agy session (e.g. 'gemini-3.8-flash-high', 'claude-sonnet-4-6').
        effort: Optional reasoning effort level ('low', 'medium', 'high', 'xhigh', 'max').
            Omit this for models whose name already encodes an effort level.
        json_schema: Optional JSON Schema object passed to agy to enforce structured output.

    Returns:
        dict | list: Parsed JSON response payload from the model.
    """
    full_prompt = (
        f"{system_prompt}\n\n{user_prompt}".strip()
        if user_prompt
        else system_prompt.strip()
    )

    cmd = [
        "agy",
        "-p",
        full_prompt,
        "--model",
        model,
        "--output-format",
        "json",
    ]
    if effort is not None:
        cmd.extend(["--effort", effort])
    if json_schema is not None:
        cmd.extend(["--json-schema", json.dumps(json_schema)])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        print(f"agy error: {stderr.decode()}")
        return {}

    envelope = json.loads(stdout.decode())
    if envelope.get("status") != "SUCCESS":
        print(f"Run failed: {envelope.get('error', 'unknown error')}")
        return {}

    structured_output = envelope.get("structured_output")
    if structured_output is not None:
        return structured_output

    raw_response = envelope.get("response", "")

    # Extract JSON content from markdown code fences or direct string
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_response)
    cleaned_response = match.group(1).strip() if match else raw_response.strip()

    try:
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Failed to parse JSON response: {e}")
        return {}
