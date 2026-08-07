"""The evaluation loop.

Responsibilities, in order:

1. build the runner from the config;
2. render the prompt (the schema's field descriptions can be injected);
3. for each case, look in the cache, otherwise call the backend;
4. score the output against the ground truth;
5. record provenance so the run means something in three weeks.

Concurrency is bounded. On a local GPU, oversubscribing does not increase
throughput — it increases latency variance and, on unified memory, swap.
"""

from __future__ import annotations

import asyncio
import platform
from datetime import UTC, datetime
from typing import Any

from docvlm_eval import __version__
from docvlm_eval.cache import CachedInference, ResultCache
from docvlm_eval.compare import score_case
from docvlm_eval.config import DEFAULT_PROMPT, Config
from docvlm_eval.corpus import Case, Corpus
from docvlm_eval.runners import Runner, build_runner
from docvlm_eval.types import CaseResult, RunResult


def render_prompt(config: Config, corpus: Corpus) -> str:
    """Fill ``{fields}`` and ``{schema}`` placeholders in the prompt template.

    Keeping the field list in one place means adding a field to the schema
    cannot leave the prompt describing the old one.
    """
    template = config.prompt or DEFAULT_PROMPT
    replacements = {
        "{fields}": corpus.schema.field_description_block(),
        "{schema}": _compact_schema(corpus.schema.json_schema()),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def _compact_schema(schema: dict[str, Any]) -> str:
    import json

    return json.dumps(schema, ensure_ascii=False, indent=2)


def resolve_scoring_policy(corpus: Corpus, config: Config) -> tuple[dict[str, float], list[str]]:
    """Merge the scoring policy declared in the schema with the config override.

    The schema is the primary source — ``critical=True`` next to the field it
    describes is where the reader will look for it. The YAML can override per
    run, which is how you ask "what if I stopped treating this field as
    critical?" without editing the schema.

    Resolving this once, in the engine, is what keeps ``report`` and ``diff``
    from quietly recomputing different numbers than ``run`` printed.
    """
    specs = corpus.schema.specs()
    weights = {name: spec.weight for name, spec in specs.items()}
    weights.update(config.weights)
    critical = {name for name, spec in specs.items() if spec.critical}
    critical.update(config.critical_fields)
    return weights, sorted(critical)


def build_runner_for(config: Config, corpus: Corpus) -> Runner:
    """Instantiate the backend named by the config."""
    kwargs: dict[str, Any] = {}
    if config.runner == "ollama":
        hosts = config.hosts()
        if hosts:
            kwargs["host"] = hosts
        if "timeout" in config.params:
            kwargs["timeout"] = float(config.params["timeout"])
    elif config.runner == "mock":
        kwargs["truths"] = {c.id: c.truth for c in corpus}
    return build_runner(config.runner, config.model, config.params, **kwargs)


async def run_config(
    corpus: Corpus,
    config: Config,
    *,
    run_name: str | None = None,
    concurrency: int = 2,
    cache: ResultCache | None = None,
    progress=None,
) -> RunResult:
    """Evaluate one config over one corpus."""
    runner = build_runner_for(config, corpus)
    prompt = render_prompt(config, corpus)
    json_schema = corpus.schema.json_schema()
    schema_hash = corpus.schema.schema_hash()
    weights, critical = resolve_scoring_policy(corpus, config)
    started = datetime.now(UTC).isoformat()

    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: dict[str, CaseResult] = {}

    async def one(case: Case) -> None:
        async with semaphore:
            results[case.id] = await _evaluate_case(
                case, corpus, config, runner, prompt, json_schema, schema_hash, cache
            )
            if progress is not None:
                progress(results[case.id])

    try:
        await asyncio.gather(*(one(case) for case in corpus))
        backend_info = await runner.describe()
    finally:
        await runner.aclose()

    return RunResult(
        name=run_name or f"{config.name}",
        corpus_name=corpus.name,
        corpus_hash=corpus.hash,
        config_name=config.name,
        config_hash=config.hash,
        field_names=list(corpus.schema.model_fields),
        cases=[results[c.id] for c in corpus if c.id in results],
        weights=weights,
        critical_fields=critical,
        provenance={
            "docvlm_eval_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "config": config.to_dict(),
            "prompt_hash": config.prompt_hash,
            "schema_hash": schema_hash,
            "corpus_hash": corpus.hash,
            "n_cases": len(corpus),
            "concurrency": concurrency,
            "backend": backend_info,
        },
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
    )


async def _evaluate_case(
    case: Case,
    corpus: Corpus,
    config: Config,
    runner: Runner,
    prompt: str,
    json_schema: dict[str, Any],
    schema_hash: str,
    cache: ResultCache | None,
) -> CaseResult:
    image = case.image_bytes()
    key = ResultCache.key(config.hash, case.id, case.image_hash(), schema_hash) if cache else ""

    hit = cache.get(key) if cache else None
    if hit is not None:
        return score_case(
            case.id,
            case.tags,
            case.truth,
            hit.data,
            corpus.schema,
            error=hit.error,
            latency_ms=hit.latency_ms,
            tokens_in=hit.tokens_in,
            tokens_out=hit.tokens_out,
            cost_usd=hit.cost_usd,
            raw_output=hit.raw,
            cached=True,
        )

    output = await runner.extract(image, prompt, json_schema, case_id=case.id)

    if cache and not output.error:
        cache.put(
            key,
            config.hash,
            case.id,
            CachedInference(
                data=output.data,
                raw=output.raw,
                latency_ms=output.latency_ms,
                tokens_in=output.tokens_in,
                tokens_out=output.tokens_out,
                cost_usd=output.cost_usd,
                error=output.error,
                meta=output.meta,
            ),
        )

    return score_case(
        case.id,
        case.tags,
        case.truth,
        output.data,
        corpus.schema,
        error=output.error,
        latency_ms=output.latency_ms,
        tokens_in=output.tokens_in,
        tokens_out=output.tokens_out,
        cost_usd=output.cost_usd,
        raw_output=output.raw,
    )
