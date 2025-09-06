"""
Модуль для разделения markdown текста на чанки по заголовкам второго уровня (##).
Каждый чанк содержит один раздел от ## до следующего ## или конца файла.
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from langchain.schema import Document


@dataclass
class ChunkerConfig:
    """Конфигурация для чанкера"""
    # Эти параметры сохраняем для совместимости, но не используем
    chunk_tokens: int = 300
    overlap_tokens: int = 25
    max_code_chunk_tokens: int = 420

    split_by_header: str = "##"  # По каким заголовкам делить
    include_header_in_chunk: bool = True  # Включать заголовок в чанк
    ignore_before_first_header: bool = True  # Игнорировать текст до первого ##


@dataclass
class ChunkMetadata:
    """Метаданные для чанка"""
    chunk_id: int
    section_title: str
    document_title: str
    source_name: str
    student_id: str
    assignment_id: str
    version: int

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует метаданные в словарь"""
        return {
            "chunk_id": self.chunk_id,
            "section_title": self.section_title,
            "document_title": self.document_title,
            "source_name": self.source_name,
            "student_id": self.student_id,
            "assignment_id": self.assignment_id,
            "version": self.version
        }


def chunk_student_markdown(
        text_md: str,
        student_id: str,
        assignment_id: str,
        version: int,
        cfg: Optional[ChunkerConfig] = None,
        source_name: Optional[str] = None
) -> List[Document]:
    """
    Разделяет markdown текст студента на чанки по заголовкам второго уровня.

    Args:
        text_md: Нормализованный markdown текст
        student_id: ID студента
        assignment_id: ID задания (dataset_id)
        version: Версия датасета
        cfg: Конфигурация чанкера
        source_name: Имя источника/документа

    Returns:
        Список Document объектов (чанков)
    """
    if cfg is None:
        cfg = ChunkerConfig()

    if not text_md:
        return []

    document_title = extract_h1_title(text_md)

    sections = split_by_h2_headers(text_md, cfg.ignore_before_first_header)

    documents = []
    for idx, section in enumerate(sections):
        if not section['content'].strip():
            continue

        metadata = ChunkMetadata(
            chunk_id=idx,
            section_title=section['title'],
            document_title=document_title or "",
            source_name=source_name or "document",
            student_id=student_id,
            assignment_id=assignment_id,
            version=version
        )

        doc = Document(
            page_content=section['content'],
            metadata=metadata.to_dict()
        )

        documents.append(doc)

    return documents


def extract_h1_title(text: str) -> Optional[str]:
    """
    Извлекает заголовок первого уровня (#) из текста.

    Args:
        text: Markdown текст

    Returns:
        Заголовок H1 или None
    """
    match = re.search(r'^#\s+(.+?)$', text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def split_by_h2_headers(text: str, ignore_before_first: bool = True) -> List[Dict[str, str]]:
    """
    Разделяет текст на секции по заголовкам второго уровня (##).

    Args:
        text: Markdown текст
        ignore_before_first: Игнорировать текст до первого ##

    Returns:
        Список секций с заголовками и содержимым
    """
    sections = []

    h2_pattern = r'^##\s+(.+?)$'

    headers = []
    for match in re.finditer(h2_pattern, text, re.MULTILINE):
        headers.append({
            'title': match.group(1).strip(),
            'start': match.start(),
            'end': match.end()
        })

    if not headers:
        if not ignore_before_first and text.strip():
            sections.append({
                'title': 'Без заголовка',
                'content': text.strip()
            })
        return sections

    if not ignore_before_first and headers[0]['start'] > 0:
        pre_text = text[:headers[0]['start']].strip()
        if pre_text:
            sections.append({
                'title': 'Введение',
                'content': pre_text
            })

    for i, header in enumerate(headers):
        section_start = header['start']
        if i < len(headers) - 1:
            section_end = headers[i + 1]['start']
        else:
            section_end = len(text)

        section_content = text[section_start:section_end].strip()

        if section_content:
            sections.append({
                'title': header['title'],
                'content': section_content
            })

    return sections


def estimate_chunk_size(text: str) -> int:
    """
    Оценивает размер чанка в символах.

    Args:
        text: Текст чанка

    Returns:
        Размер в символах
    """
    return len(text)


def get_chunk_statistics(documents: List[Document]) -> Dict[str, Any]:
    """
    Собирает статистику по чанкам.

    Args:
        documents: Список чанков

    Returns:
        Словарь со статистикой
    """
    if not documents:
        return {
            'total_chunks': 0,
            'avg_size': 0,
            'min_size': 0,
            'max_size': 0,
            'sections': []
        }

    sizes = [len(doc.page_content) for doc in documents]
    sections = [doc.metadata.get('section_title', '') for doc in documents]

    return {
        'total_chunks': len(documents),
        'avg_size': sum(sizes) // len(sizes),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'sections': sections
    }


def merge_small_chunks(
        documents: List[Document],
        min_size: int = 100
) -> List[Document]:
    """
    Объединяет слишком маленькие чанки с соседними.
    НЕ ИСПОЛЬЗУЕТСЯ в текущей реализации, так как приоритет - семантическая целостность.

    Args:
        documents: Список чанков
        min_size: Минимальный размер чанка

    Returns:
        Список чанков после объединения
    """
    return documents