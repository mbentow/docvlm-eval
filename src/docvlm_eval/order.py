"""Run order where every prefix is a fair sample of the corpus.

Stopping a benchmark early is only honest if the documents seen so far look
like the documents not yet seen. A manifest is usually sorted — by stratum, by
date, by whatever the export query did — so its first 20% can be all the easy
cases. Stopping there measures the easy cases.

``stratified_order`` shuffles inside each stratum with a fixed seed, then
interleaves the strata so that after any number of documents each stratum has
had its share, to within one document. The order is fixed *before* any result
exists, which is what keeps the sequential test valid.
"""

from __future__ import annotations

import random

from docvlm_eval.corpus import Corpus

NO_TAG = "(untagged)"


def stratum_of(tags: list[str], by: str) -> str:
    """``first`` = the first tag (exporters put the stratum there); ``all`` =
    the whole tag set; anything else = the first tag starting with that prefix."""
    if not tags:
        return NO_TAG
    if by == "first":
        return tags[0]
    if by == "all":
        return "|".join(sorted(tags))
    return next((t for t in tags if t.startswith(by)), NO_TAG)


def stratified_order(corpus: Corpus, *, by: str = "first", seed: int = 7) -> list[str]:
    strata: dict[str, list[str]] = {}
    for case in corpus:
        strata.setdefault(stratum_of(case.tags, by), []).append(case.id)
    rng = random.Random(seed)
    for name in sorted(strata):
        ids = sorted(strata[name])
        rng.shuffle(ids)
        strata[name] = ids

    total = sum(len(v) for v in strata.values())
    taken = dict.fromkeys(strata, 0)
    order: list[str] = []
    for i in range(1, total + 1):
        # The stratum furthest behind its proportional share goes next; ties by
        # name, so the order is a pure function of corpus and seed.
        name = max(
            (n for n in strata if taken[n] < len(strata[n])),
            key=lambda n: (len(strata[n]) * i / total - taken[n], n),
        )
        order.append(strata[name][taken[name]])
        taken[name] += 1
    return order
