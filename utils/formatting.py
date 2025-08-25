import re

def enforce_markdown_spacing(text: str) -> str:

    # Ensure one blank line before and after any **Heading**
    text = re.sub(r'(\*\*[^\*]+\*\*)', r'\n\1\n', text)

    # Collapse 3+ newlines into just 2
    text = re.sub(r'\n{3,}', r'\n\n', text)

    # Strip leading/trailing whitespace
    return text.strip()