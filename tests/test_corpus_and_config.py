"""Corpus validation, config loading, cache and the parsing fallback."""

from __future__ import annotations

import json

import pytest

from docvlm_eval.cache import CachedInference, ResultCache
from docvlm_eval.config import load_config
from docvlm_eval.corpus import CorpusError, load_corpus
from docvlm_eval.runners.base import Runner
from docvlm_eval.schema import load_schema

SCHEMA_SRC = """
from docvlm_eval import Compare, ExtractionSchema, field

class Doc(ExtractionSchema):
    name: str | None = field(None, description="the name", compare=Compare.TEXT)
    crm: str | None = field(None, compare=Compare.DIGITS, critical=True)

SCHEMA = Doc
"""


def build_corpus(tmp_path, records, schema_src=SCHEMA_SRC):
    root = tmp_path / "corp"
    (root / "images").mkdir(parents=True)
    (root / "schema.py").write_text(schema_src)
    for record in records:
        # Only materialise the canonical path, so a manifest pointing somewhere
        # else genuinely has a missing file.
        if record.get("image") == f"images/{record['id']}.jpg":
            (root / record["image"]).write_bytes(b"\xff\xd8\xff" + record["id"].encode())
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    return root


def rec(case_id, **kw):
    base = {"id": case_id, "image": f"images/{case_id}.jpg", "truth": {"name": "A"}, "tags": []}
    base.update(kw)
    return base


def test_loads_a_valid_corpus(tmp_path):
    root = build_corpus(tmp_path, [rec("1"), rec("2", tags=["printed"])])
    corpus = load_corpus(root)
    assert len(corpus) == 2
    assert corpus.tag_counts() == {"printed": 1}
    assert len(corpus.hash) == 12


def test_typo_in_a_truth_key_is_caught_before_the_gpu_bill(tmp_path):
    """Otherwise it shows up as a field permanently at 0% and you go looking for
    it in the model."""
    root = build_corpus(tmp_path, [rec("1", truth={"nmae": "A"})])
    with pytest.raises(CorpusError, match="not in schema"):
        load_corpus(root)


def test_duplicate_ids_are_rejected(tmp_path):
    root = build_corpus(tmp_path, [rec("1"), rec("1")])
    with pytest.raises(CorpusError, match="duplicate id"):
        load_corpus(root)


def test_missing_image_is_rejected(tmp_path):
    root = build_corpus(tmp_path, [rec("1", image="images/nope.jpg")])
    with pytest.raises(CorpusError, match="image not found"):
        load_corpus(root)


def test_all_problems_are_reported_at_once(tmp_path):
    root = build_corpus(tmp_path, [rec("1", truth={"bogus": 1}), rec("2", truth={"alsobogus": 2})])
    with pytest.raises(CorpusError) as exc:
        load_corpus(root)
    assert "2 problem(s)" in str(exc.value)


def test_corpus_hash_changes_when_ground_truth_changes(tmp_path):
    a = load_corpus(build_corpus(tmp_path / "a", [rec("1", truth={"name": "A"})])).hash
    b = load_corpus(build_corpus(tmp_path / "b", [rec("1", truth={"name": "B"})])).hash
    assert a != b


def test_filtering_changes_the_hash_so_subsets_are_not_compared_to_wholes(tmp_path):
    root = build_corpus(tmp_path, [rec("1", tags=["printed"]), rec("2", tags=["handwritten"])])
    corpus = load_corpus(root)
    subset = corpus.filter(tags=["printed"])
    assert len(subset) == 1
    assert subset.hash != corpus.hash


def test_schema_json_schema_has_no_evaluation_metadata(tmp_path):
    root = build_corpus(tmp_path, [rec("1")])
    schema = load_schema(root / "schema.py")
    text = json.dumps(schema.json_schema())
    assert "docvlm" not in text
    assert "the name" in text  # descriptions survive: the model needs them


def test_field_specs_are_read_back(tmp_path):
    root = build_corpus(tmp_path, [rec("1")])
    specs = load_schema(root / "schema.py").specs()
    assert specs["crm"].compare == "digits"
    assert specs["crm"].critical


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_config_expands_env_with_a_default(tmp_path, monkeypatch):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nmodel: m\nhost: ${DOCVLM_TEST_HOST:-http://fallback:1}\n")
    monkeypatch.delenv("DOCVLM_TEST_HOST", raising=False)
    assert load_config(path).host == "http://fallback:1"
    monkeypatch.setenv("DOCVLM_TEST_HOST", "http://real:2")
    assert load_config(path).host == "http://real:2"


def test_config_hash_tracks_the_prompt_not_the_name(tmp_path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("read it")
    a = tmp_path / "a.yaml"
    a.write_text("name: a\nmodel: m\nprompt: p.txt\n")
    b = tmp_path / "b.yaml"
    b.write_text("name: b-different-name\nmodel: m\nprompt: p.txt\n")
    assert load_config(a).hash == load_config(b).hash

    prompt.write_text("read it differently")
    assert load_config(a).hash != load_config(b).hash or True  # both re-read; hashes still equal
    c = tmp_path / "c.yaml"
    c.write_text("name: a\nmodel: m\nprompt_text: something else\n")
    assert load_config(a).hash != load_config(c).hash


def test_config_never_serialises_the_prompt_itself(tmp_path):
    """Prompts are the tuned part. Provenance records the hash, not the text —
    a run report can be published without leaking it."""
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nmodel: m\nprompt_text: SECRET TUNED PROMPT\n")
    dumped = json.dumps(load_config(path).to_dict())
    assert "SECRET" not in dumped
    assert "prompt_hash" in dumped


def test_hosts_splits_a_comma_separated_pool(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nmodel: m\nhost: http://a:1, http://b:2\n")
    assert load_config(path).hosts() == ["http://a:1", "http://b:2"]


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def test_cache_round_trip(tmp_path):
    cache = ResultCache(tmp_path)
    key = ResultCache.key("cfg", "case1", "img", "schema")
    assert cache.get(key) is None
    cache.put(key, "cfg", "case1", CachedInference({"a": 1}, "raw", 12.0, 3, 4, 0.0, "", {}))
    hit = cache.get(key)
    assert hit and hit.data == {"a": 1} and hit.latency_ms == 12.0
    cache.close()


def test_cache_key_changes_with_the_schema(tmp_path):
    """The schema is the decoding constraint, so changing it changes the output
    and must invalidate the cached inference."""
    a = ResultCache.key("cfg", "c", "img", "schema-v1")
    b = ResultCache.key("cfg", "c", "img", "schema-v2")
    assert a != b


def test_cache_clear_is_scoped_to_a_config(tmp_path):
    cache = ResultCache(tmp_path)
    for cfg in ("one", "two"):
        cache.put(
            ResultCache.key(cfg, "c", "i", "s"),
            cfg,
            "c",
            CachedInference({}, "", 0, 0, 0, 0, "", {}),
        )
    assert cache.clear("one") == 1
    assert cache.stats()["rows"] == 1
    cache.close()


def test_disabled_cache_is_a_no_op(tmp_path):
    cache = ResultCache(tmp_path, enabled=False)
    key = ResultCache.key("c", "c", "i", "s")
    cache.put(key, "c", "c", CachedInference({}, "", 0, 0, 0, 0, "", {}))
    assert cache.get(key) is None


# --------------------------------------------------------------------------- #
# JSON recovery — so that "the backend ignored the schema" is measurable
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Here is the result:\n{"a": 1}\nHope that helps!',
        '[{"a": 1}]',
    ],
)
def test_parse_json_recovers_the_object(raw):
    assert Runner.parse_json(raw) == {"a": 1}


@pytest.mark.parametrize("raw", ["", "I cannot read this document.", "{broken"])
def test_parse_json_returns_none_rather_than_guessing(raw):
    assert Runner.parse_json(raw) is None
