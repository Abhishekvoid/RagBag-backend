import re

def enforce_markdown_spacing(text: str) -> str:
    text = re.sub(r'(\*\*[^\*]+\*\*)', r'\n\1\n', text)

    text = re.sub(r'\n{3,}', r'\n\n', text)

    return text.strip()