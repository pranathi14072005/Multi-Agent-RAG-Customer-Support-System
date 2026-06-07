"""
kb/knowledge_base.py
--------------------
Knowledge Base manager backed by ChromaDB.

Two collections:
  - knowledge_base     : curated support articles and documentation
  - resolved_tickets   : indexed resolved tickets for similar-case retrieval

Features:
  - Ingest files, URLs, directories, or raw text
  - Semantic search with category filtering
  - Similar-case retrieval (for Tier-2 agents)
  - Version-aware document updates
  - Staleness detection
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import KBDocument, SupportCategory, SupportTicket
from settings import Settings

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Dual-collection ChromaDB knowledge base for the support system.

    kb_collection      → polished support articles (used by Tier-1 + Tier-2)
    tickets_collection → indexed resolved tickets (used by Tier-2 similar-case lookup)
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.embeddings = OllamaEmbeddings(
            model=self.settings.embedding_model,
            base_url=self.settings.ollama_base_url,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self._kb_store: Optional[Chroma] = self._load_or_create(
            self.settings.chroma_kb_collection
        )
        self._tickets_store: Optional[Chroma] = self._load_or_create(
            self.settings.chroma_tickets_collection
        )
        logger.info("KnowledgeBase initialised. Persist dir: %s", self.settings.chroma_persist_dir)

    # ── Ingestion ──────────────────────────────────────────────────────

    def ingest_file(self, path: str, category: SupportCategory = SupportCategory.GENERAL,
                    tags: Optional[list[str]] = None) -> int:
        """Ingest a file (PDF, txt, md) or directory into the KB."""
        docs = self._load_source(path)
        return self._add_to_kb(docs, source=path, category=category, tags=tags or [])

    def ingest_url(self, url: str, category: SupportCategory = SupportCategory.GENERAL,
                   tags: Optional[list[str]] = None) -> int:
        docs = WebBaseLoader(url).load()
        return self._add_to_kb(docs, source=url, category=category, tags=tags or [])

    def ingest_text(self, text: str, title: str, source: str = "manual",
                    category: SupportCategory = SupportCategory.GENERAL,
                    tags: Optional[list[str]] = None) -> int:
        doc = Document(
            page_content=text,
            metadata={"title": title, "source": source, "category": category.value},
        )
        return self._add_to_kb([doc], source=source, category=category, tags=tags or [])

    def ingest_kb_document(self, kb_doc: KBDocument) -> int:
        """Ingest a KBDocument object directly."""
        lc_doc = kb_doc.to_langchain_document()
        return self._add_to_kb([lc_doc], source=kb_doc.source,
                                category=kb_doc.category, tags=kb_doc.tags)

    def index_resolved_ticket(self, ticket: SupportTicket) -> None:
        """Index a resolved ticket into the tickets collection for similar-case retrieval."""
        if not ticket.resolution_summary:
            logger.warning("Ticket %s has no resolution summary, skipping index.", ticket.ticket_id)
            return

        content = (
            f"Issue: {ticket.subject}\n"
            f"Category: {ticket.category.value}\n"
            f"Resolution: {ticket.resolution_summary}\n"
        )
        # Append key message excerpts
        user_msgs = [m.content for m in ticket.messages if m.role == "user"]
        if user_msgs:
            content += f"\nCustomer description: {user_msgs[0][:300]}"

        doc = Document(
            page_content=content,
            metadata={
                "ticket_id": ticket.ticket_id,
                "category": ticket.category.value,
                "subject": ticket.subject,
                "source_type": "ticket",
                "tags": ",".join(ticket.tags),
            },
        )
        if self._tickets_store is None:
            self._tickets_store = Chroma(
                persist_directory=self.settings.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name=self.settings.chroma_tickets_collection,
            )
        self._tickets_store.add_documents([doc])
        self._tickets_store.persist()
        logger.info("Indexed resolved ticket %s.", ticket.ticket_id)

    # ── Retrieval ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[SupportCategory] = None,
    ) -> list[Document]:
        """Semantic search over the KB collection."""
        if not self.has_kb_documents():
            return []
        k = top_k or self.settings.retriever_top_k
        kwargs: dict = {"k": k}
        if category_filter:
            kwargs["filter"] = {"category": category_filter.value}
        try:
            return self._kb_store.similarity_search(query, **kwargs)
        except Exception as exc:
            logger.warning("KB search failed: %s", exc)
            return []

    def search_similar_cases(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[SupportCategory] = None,
    ) -> list[Document]:
        """Search resolved tickets for similar past cases."""
        if not self._tickets_store:
            return []
        kwargs: dict = {"k": top_k}
        if category_filter:
            kwargs["filter"] = {"category": category_filter.value}
        try:
            return self._tickets_store.similarity_search(query, **kwargs)
        except Exception as exc:
            logger.warning("Ticket store search failed: %s", exc)
            return []

    def format_context(self, docs: list[Document]) -> str:
        """Format retrieved docs into a single context string for the LLM prompt."""
        if not docs:
            return "No relevant knowledge base articles found."
        parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", f"Article {i}")
            source = doc.metadata.get("source", "")
            parts.append(f"[{i}] {title}\nSource: {source}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def format_similar_cases(self, docs: list[Document]) -> str:
        if not docs:
            return "No similar resolved cases found."
        parts = []
        for i, doc in enumerate(docs, 1):
            subject = doc.metadata.get("subject", f"Case {i}")
            tid = doc.metadata.get("ticket_id", "unknown")
            parts.append(f"[Case {i}] {subject} (Ticket: {tid})\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # ── State Checks ───────────────────────────────────────────────────

    def has_kb_documents(self) -> bool:
        if self._kb_store is None:
            return False
        try:
            return self._kb_store._collection.count() > 0
        except Exception:
            return False

    def kb_document_count(self) -> int:
        if self._kb_store is None:
            return 0
        try:
            return self._kb_store._collection.count()
        except Exception:
            return 0

    def resolved_ticket_count(self) -> int:
        if self._tickets_store is None:
            return 0
        try:
            return self._tickets_store._collection.count()
        except Exception:
            return 0

    def clear_kb(self) -> None:
        if self._kb_store:
            self._kb_store.delete_collection()
            self._kb_store = None
            logger.warning("KB collection cleared.")

    def clear_ticket_index(self) -> None:
        if self._tickets_store:
            self._tickets_store.delete_collection()
            self._tickets_store = None
            logger.warning("Ticket index cleared.")

    # ── Private Helpers ────────────────────────────────────────────────

    def _load_or_create(self, collection_name: str) -> Optional[Chroma]:
        persist_dir = self.settings.chroma_persist_dir
        if persist_dir and Path(persist_dir).exists():
            try:
                store = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=collection_name,
                )
                logger.info("Loaded ChromaDB collection '%s'.", collection_name)
                return store
            except Exception as exc:
                logger.warning("Could not load collection '%s': %s", collection_name, exc)
        return None

    def _add_to_kb(self, docs: list[Document], source: str,
                   category: SupportCategory, tags: list[str]) -> int:
        if not docs:
            logger.warning("No documents to ingest from source: %s", source)
            return 0
        chunks = self.splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata.setdefault("category", category.value)
            chunk.metadata.setdefault("tags", ",".join(tags))
            chunk.metadata.setdefault("source", source)

        if self._kb_store is None:
            self._kb_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.settings.chroma_persist_dir,
                collection_name=self.settings.chroma_kb_collection,
            )
        else:
            self._kb_store.add_documents(chunks)
        self._kb_store.persist()
        logger.info("Ingested %d chunks from '%s'.", len(chunks), source)
        return len(chunks)

    @staticmethod
    def _load_source(source: str) -> list[Document]:
        if os.path.isdir(source):
            docs = []
            for root, _, files in os.walk(source):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith((".txt", ".md")):
                        try:
                            docs.extend(TextLoader(file_path, encoding="utf-8").load())
                        except Exception as e:
                            logger.warning("Failed to load %s: %s", file_path, e)
                    elif file.endswith(".pdf"):
                        try:
                            docs.extend(PyPDFLoader(file_path).load())
                        except Exception as e:
                            logger.warning("Failed to load %s: %s", file_path, e)
            return docs
        elif source.endswith(".pdf"):
            return PyPDFLoader(source).load()
        else:
            return TextLoader(source, encoding="utf-8").load()
