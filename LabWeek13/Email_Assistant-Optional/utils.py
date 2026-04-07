# ================================
# Imports
# ================================

# --- Standard library ---
import os
import json
from html import escape
from datetime import datetime  # keep if used later
from urllib.parse import urljoin

# --- Third-party ---
import requests
import pandas as pd
from IPython.display import display, HTML

import base64
from typing import Any


# ================================
# Helpers
# ================================
def print_html(content: Any, title: str | None = None, is_image: bool = False):
    """
    Pretty-print inside a styled card.
    - If is_image=True and content is a string: treat as image path/URL and render <img>.
    - If content is a pandas DataFrame/Series: render as an HTML table.
    - Otherwise (strings/otros): show as code/text in <pre><code>.
    """
    try:
        from html import escape as _escape
    except ImportError:
        _escape = lambda x: x

    def image_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    # Render content
    if is_image and isinstance(content, str):
        b64 = image_to_base64(content)
        rendered = f'<img src="data:image/png;base64,{b64}" alt="Image" style="max-width:100%; height:auto; border-radius:8px;">'
    elif isinstance(content, pd.DataFrame):
        rendered = content.to_html(classes="pretty-table", index=False, border=0, escape=False)
    elif isinstance(content, pd.Series):
        rendered = content.to_frame().to_html(classes="pretty-table", border=0, escape=False)
    elif isinstance(content, str):
        rendered = f"<pre><code>{_escape(content)}</code></pre>"
    else:
        rendered = f"<pre><code>{_escape(str(content))}</code></pre>"

    css = """
    <style>
    .pretty-card{
      font-family: ui-sans-serif, system-ui;
      border: 2px solid transparent;
      border-radius: 14px;
      padding: 14px 16px;
      margin: 10px 0;
      background: linear-gradient(#fff, #fff) padding-box,
                  linear-gradient(135deg, #3b82f6, #9333ea) border-box;
      color: #111;
      box-shadow: 0 4px 12px rgba(0,0,0,.08);
    }
    .pretty-title{
      font-weight:700;
      margin-bottom:8px;
      font-size:14px;
      color:#111;
    }
    /* 🔒 Solo afecta lo DENTRO de la tarjeta */
    .pretty-card pre, 
    .pretty-card code {
      background: #f3f4f6;
      color: #111;
      padding: 8px;
      border-radius: 8px;
      display: block;
      overflow-x: auto;
      font-size: 13px;
      white-space: pre-wrap;
    }
    .pretty-card img { max-width: 100%; height: auto; border-radius: 8px; }
    .pretty-card table.pretty-table {
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      color: #111;
    }
    .pretty-card table.pretty-table th, 
    .pretty-card table.pretty-table td {
      border: 1px solid #e5e7eb;
      padding: 6px 8px;
      text-align: left;
    }
    .pretty-card table.pretty-table th { background: #f9fafb; font-weight: 600; }
    </style>
    """

    title_html = f'<div class="pretty-title">{title}</div>' if title else ""
    card = f'<div class="pretty-card">{title_html}{rendered}</div>'
    display(HTML(css + card))

def pretty_display(title: str, response: requests.Response):
    """Render an HTTP response in a styled block; returns parsed content (JSON if possible)."""
    status = response.status_code
    try:
        content = response.json()
        body = json.dumps(content, indent=2)
    except Exception:
        content = response.text
        body = content

    html = f"""
    <div style='border:1px solid #ccc; border-left:5px solid #007bff; padding:10px; margin:10px 0; background:#f9f9f9; color:#000;'>
        <strong style='color:#007bff'>{escape(title)}:</strong>
        <span style='color:{"green" if status == 200 else "red"}'> Status {status}</span>
        <pre style='font-size:12px; margin-top:10px; white-space:pre-wrap; color:#000;'>{escape(body)}</pre>
    </div>
    """
    display(HTML(html))
    return content

def pretty_print_chat_completion(response):
    def format_json(data):
        try:
            return json.dumps(data, indent=2)
        except:
            return str(data)

    steps_html = ""
    tool_sequence = []  # ← Track tool names
    choice = response.choices[0]
    intermediate_messages = getattr(choice, "intermediate_messages", [])

    for step in intermediate_messages:
        # Step: LLM decision to call a tool
        if hasattr(step, "tool_calls") and step.tool_calls:
            for call in step.tool_calls:
                tool_name = call.function.name
                tool_sequence.append(tool_name)
                args = json.loads(call.function.arguments)
                steps_html += f"""
                <div style="border-left: 4px solid #444; margin: 10px 0; padding: 10px; background: #f0f0f0;">
                    <strong style="color:#222;">🧠 LLM Action:</strong> <code>{tool_name}</code>
                    <pre style="color:#000; font-size:13px;">{format_json(args)}</pre>
                </div>
                """
        # Step: tool response
        elif isinstance(step, dict) and step.get("role") == "tool":
            tool_name = step.get("name")
            tool_output = step.get("content")
            try:
                parsed_output = json.loads(tool_output)
            except:
                parsed_output = tool_output
            steps_html += f"""
            <div style="border-left: 4px solid #007bff; margin: 10px 0; padding: 10px; background: #eef6ff;">
                <strong style="color:#222;">🔧 Tool Response:</strong> <code>{tool_name}</code>
                <pre style="color:#000; font-size:13px;">{format_json(parsed_output)}</pre>
            </div>
            """

    # Final assistant message
    final_msg = choice.message.content
    steps_html += f"""
    <div style="border-left: 4px solid #28a745; margin: 20px 0; padding: 10px; background: #eafbe7;">
        <strong style="color:#222;">✅ Final Assistant Message:</strong>
        <p style="color:#000;">{final_msg}</p>
    </div>
    """

    # Tool sequence summary
    if tool_sequence:
        arrow_sequence = " → ".join(tool_sequence)
        steps_html += f"""
        <div style="border-left: 4px solid #666; margin: 20px 0; padding: 10px; background: #f8f9fa;">
            <strong style="color:#222;">🧭 Tool Sequence:</strong>
            <p style="color:#000;">{arrow_sequence}</p>
        </div>
        """

    display(HTML(steps_html))