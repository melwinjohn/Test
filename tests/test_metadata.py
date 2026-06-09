from pathlib import Path

from rag_pipeline.loaders import build_document_metadata, load_manifest, load_metadata_sidecar
from rag_pipeline.metadata import (
    build_chunk_metadata,
    detect_pii_categories,
    infer_document_type,
    merge_metadata,
    metadata_matches_filters,
)
from rag_pipeline.vector_store import MetadataFilter


def test_infer_document_type_from_path():
    assert infer_document_type("data/invoices/acme-001.txt") is not None
    assert infer_document_type("data/invoices/acme-001.txt").value == "invoice"
    assert infer_document_type("contracts/msa.txt").value == "contract"


def test_merge_metadata_layers():
    merged = merge_metadata(
        {"tenant_id": "acme", "contains_pii": False},
        {"document_type": "invoice", "contains_pii": True},
    )
    assert merged["tenant_id"] == "acme"
    assert merged["document_type"] == "invoice"
    assert merged["contains_pii"] is True


def test_build_chunk_metadata_inherits_document_fields():
    metadata = build_chunk_metadata(
        {
            "document_type": "invoice",
            "contains_pii": True,
            "pii_level": "high",
            "tenant_id": "acme-corp",
        },
        chunk_index=0,
        chunk_count=2,
        content="Contact john.smith@acme.com for payment.",
        detect_pii=True,
    )
    assert metadata["document_type"] == "invoice"
    assert metadata["chunk_index"] == 0
    assert metadata["chunk_count"] == 2
    assert metadata["contains_pii"] is True
    assert "email" in metadata["detected_pii_categories"]


def test_detect_pii_categories():
    text = "Reach us at jane.doe@example.com or call 555-123-4567."
    categories = detect_pii_categories(text)
    assert "email" in categories
    assert "phone" in categories


def test_metadata_matches_filters():
    metadata = {
        "document_type": "invoice",
        "contains_pii": True,
        "pii_level": "high",
        "tenant_id": "acme-corp",
        "tags": ["billing"],
    }
    assert metadata_matches_filters(metadata, document_type="invoice")
    assert not metadata_matches_filters(metadata, exclude_pii=True)
    assert not metadata_matches_filters(metadata, pii_level_max="low")
    assert metadata_matches_filters(metadata, tags=["billing"])


def test_metadata_filter_sql_clauses():
    metadata_filter = MetadataFilter(
        document_type="contract",
        exclude_pii=True,
        pii_level_max="low",
        tags=["legal"],
    )
    clauses, params = metadata_filter.to_sql()
    assert "metadata->>'document_type' = %s" in clauses
    assert "contains_pii" in " ".join(clauses)
    assert "contract" in params
    assert ["legal"] in params or "legal" in params


def test_load_metadata_sidecar(tmp_path: Path):
    doc = tmp_path / "invoice.txt"
    doc.write_text("invoice body", encoding="utf-8")
    sidecar = tmp_path / "invoice.txt.meta.json"
    sidecar.write_text('{"document_type": "invoice", "contains_pii": true}', encoding="utf-8")

    loaded = load_metadata_sidecar(doc)
    assert loaded["document_type"] == "invoice"
    assert loaded["contains_pii"] is True


def test_build_document_metadata_from_sidecar_and_manifest(tmp_path: Path):
    doc = tmp_path / "contracts" / "msa.txt"
    doc.parent.mkdir(parents=True)
    doc.write_text("contract body", encoding="utf-8")
    sidecar = doc.with_name("msa.txt.meta.json")
    sidecar.write_text('{"document_id": "MSA-1", "pii_level": "low"}', encoding="utf-8")

    metadata = build_document_metadata(
        doc,
        sidecar=load_metadata_sidecar(doc),
        manifest_entry={"tenant_id": "gamma"},
    )
    assert metadata["document_type"] == "contract"
    assert metadata["document_id"] == "MSA-1"
    assert metadata["tenant_id"] == "gamma"


def test_load_manifest_dict_and_list(tmp_path: Path):
    dict_manifest = tmp_path / "dict.json"
    dict_manifest.write_text('{"a.txt": {"document_type": "invoice"}}', encoding="utf-8")
    assert load_manifest(dict_manifest)["a.txt"]["document_type"] == "invoice"

    list_manifest = tmp_path / "list.json"
    list_manifest.write_text(
        '[{"source": "b.txt", "document_type": "contract"}]',
        encoding="utf-8",
    )
    assert load_manifest(list_manifest)["b.txt"]["document_type"] == "contract"
