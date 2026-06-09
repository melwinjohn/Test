from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    INVOICE = "invoice"
    CONTRACT = "contract"
    CORRESPONDENCE = "correspondence"
    REPORT = "report"
    OTHER = "other"


class PiiLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    HIGH = "high"
    RESTRICTED = "restricted"


PII_LEVEL_RANK = {
    PiiLevel.NONE: 0,
    PiiLevel.LOW: 1,
    PiiLevel.HIGH: 2,
    PiiLevel.RESTRICTED: 3,
}


class PiiCategory(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    TAX_ID = "tax_id"
    BANK_ACCOUNT = "bank_account"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    PERSON_NAME = "person_name"
    DATE_OF_BIRTH = "date_of_birth"


PII_PATTERNS: dict[PiiCategory, re.Pattern[str]] = {
    PiiCategory.EMAIL: re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    PiiCategory.PHONE: re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    PiiCategory.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    PiiCategory.TAX_ID: re.compile(r"\b\d{2}-\d{7}\b"),
    PiiCategory.BANK_ACCOUNT: re.compile(r"\b(?:account|acct)[#:\s]*\d{6,17}\b", re.IGNORECASE),
    PiiCategory.CREDIT_CARD: re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    PiiCategory.ADDRESS: re.compile(
        r"\b\d{1,5}\s+\w+(?:\s+\w+){0,3}\s(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive)\b",
        re.IGNORECASE,
    ),
    PiiCategory.DATE_OF_BIRTH: re.compile(
        r"\b(?:DOB|Date of Birth)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        re.IGNORECASE,
    ),
}


class DocumentMetadata(BaseModel):
    """Document-level metadata applied to every chunk from a source."""

    file_name: str | None = None
    file_path: str | None = None
    document_type: DocumentType = DocumentType.OTHER
    document_id: str | None = None
    tenant_id: str | None = None
    contains_pii: bool = False
    pii_level: PiiLevel = PiiLevel.NONE
    pii_categories: list[str] = Field(default_factory=list)
    access_tier: str | None = None
    tags: list[str] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)

    @field_validator("pii_categories", "tags", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DocumentMetadata:
        if not data:
            return cls()
        known = cls.model_fields.keys()
        doc_fields = {key: data[key] for key in data if key in known}
        custom = {key: value for key, value in data.items() if key not in known}
        if custom:
            doc_fields["custom"] = {**doc_fields.get("custom", {}), **custom}
        return cls.model_validate(doc_fields)


class ChunkMetadata(DocumentMetadata):
    """Chunk-level metadata: document fields plus chunk provenance."""

    chunk_index: int = 0
    chunk_count: int | None = None
    char_count: int = 0
    chunking: str = "semantic"
    detected_pii_categories: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def merge_metadata(*layers: dict[str, Any] | DocumentMetadata | None) -> dict[str, Any]:
    """Merge metadata layers left-to-right; later layers override earlier ones."""
    merged: dict[str, Any] = {}
    for layer in layers:
        if layer is None:
            continue
        if isinstance(layer, (DocumentMetadata, ChunkMetadata)):
            data = layer.to_dict()
        else:
            data = dict(layer)
        for key, value in data.items():
            if key == "custom" and isinstance(value, dict):
                merged.setdefault("custom", {})
                merged["custom"].update(value)
            elif value is not None:
                merged[key] = value
    return merged


def build_chunk_metadata(
    document_metadata: dict[str, Any],
    *,
    chunk_index: int,
    chunk_count: int,
    content: str,
    file_name: str | None = None,
    file_path: str | None = None,
    detect_pii: bool = False,
) -> dict[str, Any]:
    detected: list[str] = []
    if detect_pii:
        detected = detect_pii_categories(content)

    base = merge_metadata(
        document_metadata,
        {
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "char_count": len(content),
            "chunking": "semantic",
            "file_name": file_name,
            "file_path": file_path,
            "detected_pii_categories": detected,
        },
    )

    if detected and not base.get("contains_pii"):
        base["contains_pii"] = True

    if detected:
        declared = set(base.get("pii_categories", []))
        base["pii_categories"] = sorted(declared.union(detected))

    return ChunkMetadata.model_validate(base).to_dict()


def detect_pii_categories(text: str) -> list[str]:
    """Scan chunk text for common PII patterns."""
    found: list[str] = []
    for category, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found.append(category.value)
    return found


def infer_document_type(path: str) -> DocumentType | None:
    """Infer document type from file path segments or filename."""
    lowered = path.lower()
    hints = {
        DocumentType.INVOICE: ("invoice", "invoices", "bill", "billing"),
        DocumentType.CONTRACT: ("contract", "contracts", "agreement", "msa"),
        DocumentType.CORRESPONDENCE: ("letter", "email", "correspondence", "memo"),
        DocumentType.REPORT: ("report", "reports", "summary"),
    }
    for doc_type, keywords in hints.items():
        if any(keyword in lowered for keyword in keywords):
            return doc_type
    return None


def metadata_matches_filters(
    metadata: dict[str, Any],
    *,
    document_type: str | None = None,
    pii_level_max: PiiLevel | None = None,
    exclude_pii: bool = False,
    tenant_id: str | None = None,
    access_tier: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    if document_type and metadata.get("document_type") != document_type:
        return False
    if tenant_id and metadata.get("tenant_id") != tenant_id:
        return False
    if access_tier and metadata.get("access_tier") != access_tier:
        return False
    if exclude_pii and metadata.get("contains_pii"):
        return False
    if pii_level_max is not None:
        chunk_level = PiiLevel(metadata.get("pii_level", PiiLevel.NONE.value))
        if PII_LEVEL_RANK[chunk_level] > PII_LEVEL_RANK[pii_level_max]:
            return False
    if tags:
        chunk_tags = set(metadata.get("tags", []))
        if not chunk_tags.intersection(tags):
            return False
    return True
