"""
Модуль нормализации markdown текста для RAG системы.
Подготавливает текст для последующего чанкинга.
"""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class NormalizationConfig:
    """Конфигурация для нормализации текста"""
    remove_extra_whitespace: bool = True
    normalize_lists: bool = True
    clean_code_blocks: bool = True
    remove_html_tags: bool = True
    fix_encoding_issues: bool = True


def normalize_markdown(text: str, config: Optional[NormalizationConfig] = None) -> str:
    """
    Нормализует markdown текст для последующей обработки.

    Args:
        text: Исходный markdown текст
        config: Конфигурация нормализации

    Returns:
        Нормализованный текст
    """
    if config is None:
        config = NormalizationConfig()

    if not text:
        return ""

    if config.fix_encoding_issues:
        text = fix_encoding_issues(text)

    if config.remove_html_tags:
        text = remove_html_tags(text)

    text = normalize_line_breaks(text)

    if config.clean_code_blocks:
        text = clean_code_blocks(text)

    if config.normalize_lists:
        text = normalize_lists(text)

    if config.remove_extra_whitespace:
        text = remove_extra_whitespace(text)

    text = remove_trailing_spaces(text)

    text = limit_empty_lines(text)

    return text.strip()


def fix_encoding_issues(text: str) -> str:
    """Исправляет типичные проблемы с кодировкой"""
    replacements = {
        'РІ': 'в',
        'вЂ"': '—',
        'вЂ™': "'",
        'вЂњ': '"',
        'вЂќ': '"',
        'вЂў': '•',
        'вЂ¦': '…',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def remove_html_tags(text: str) -> str:
    """Удаляет HTML теги из текста"""
    code_blocks = []

    def save_code(match):
        code_blocks.append(match.group(1))
        return f"```\n{match.group(1)}\n```"

    text = re.sub(r'<code>(.*?)</code>', save_code, text, flags=re.DOTALL)
    text = re.sub(r'<pre>(.*?)</pre>', save_code, text, flags=re.DOTALL)

    text = re.sub(r'<[^>]+>', '', text)

    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")

    return text


def normalize_line_breaks(text: str) -> str:
    """Нормализует переносы строк (Unix-style)"""
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    return text


def clean_code_blocks(text: str) -> str:
    """Очищает код-блоки от лишних пробелов"""

    def clean_block(match):
        lang = match.group(1) or ''
        code = match.group(2)
        code = code.strip()
        return f"```{lang}\n{code}\n```"

    text = re.sub(r'```(\w*)\n?(.*?)\n?```', clean_block, text, flags=re.DOTALL)

    return text


def normalize_lists(text: str) -> str:
    """Нормализует маркированные и нумерованные списки"""
    lines = text.split('\n')
    result = []

    for line in lines:
        if re.match(r'^\s*[\*\+]\s+', line):
            line = re.sub(r'^(\s*)[\*\+]\s+', r'\1- ', line)

        if re.match(r'^\s*\d+\.\s+', line):
            line = re.sub(r'^(\s*\d+\.)\s+', r'\1 ', line)

        result.append(line)

    return '\n'.join(result)


def remove_extra_whitespace(text: str) -> str:
    """Убирает множественные пробелы и табуляции (кроме код-блоков)"""
    code_blocks = []

    def save_code(match):
        idx = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{idx}__"

    text = re.sub(r'```.*?```', save_code, text, flags=re.DOTALL)

    lines = text.split('\n')
    result = []
    for line in lines:
        if not line.strip().startswith('__CODE_BLOCK_'):
            indent = len(line) - len(line.lstrip())
            content = line.lstrip()
            content = re.sub(r'\t', ' ', content)
            content = re.sub(r' +', ' ', content)
            line = ' ' * indent + content
        result.append(line)
    text = '\n'.join(result)

    for idx, block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{idx}__", block)

    return text


def remove_trailing_spaces(text: str) -> str:
    """Убирает пробелы в конце строк"""
    lines = text.split('\n')
    return '\n'.join(line.rstrip() for line in lines)


def limit_empty_lines(text: str) -> str:
    """Ограничивает количество подряд идущих пустых строк (максимум 2)"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def extract_title_from_markdown(text: str) -> Optional[str]:
    """
    Извлекает заголовок H1 из markdown текста.

    Args:
        text: Markdown текст

    Returns:
        Заголовок H1 или None, если не найден
    """
    match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None