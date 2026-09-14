"""Optional LLM-backed hypothesis reasoning (Anthropic SDK).

The deterministic ``HypothesisGenerator`` never depends on this module. When
enabled, the reasoner receives the diagnosis as a numbered list of *facts* and
the deterministic hypotheses, and may (a) re-rank them and (b) propose a few
additional hypotheses. Guardrails:

* every proposed hypothesis must cite at least one provided fact id; uncited or
  fabricated evidence is rejected;
* proposed confidence is capped (default 0.7) below what strong deterministic
  evidence produces;
* the reasoner can never decide KEEP: only the Experiment Judge establishes
  predictive value;
* any API failure, refusal, or malformed reply leaves the hypotheses unchanged.

Install with ``pip install "automl-py[llm]"`` and authenticate the Anthropic client
(``ANTHROPIC_API_KEY`` or ``ant auth login``).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

import numpy as np

from .hypotheses import Diagnosis, Hypothesis

log = logging.getLogger("automl_py")

ALLOWED_KINDS = {"existing", "derivable", "external"}

SYSTEM_PROMPT = """You are the hypothesis reasoner inside a predictive-discovery system.
You receive numbered FACTS about a fitted model's remaining error and the deterministic hypotheses already generated.
Your job: propose up to {max_new} additional, testable hypotheses about information that could explain the remaining error, and re-rank the full list.
Rules:
- Every hypothesis must cite fact ids from the FACTS list as its evidence. Do not invent numbers, columns, or observations.
- A hypothesis names search concepts a data catalog could be searched with.
- kind is one of: existing (likely already in the enterprise), derivable (engineered from current columns), external (third-party data).
- Never claim a hypothesis is proven; experiments decide.
Reply with JSON only, matching:
{{"rerank": ["H-001", ...], "new": [{{"statement": "...", "evidence_ids": ["F1"], "search_concepts": ["..."], "kind": "external", "confidence": 0.5}}]}}"""


class AnthropicHypothesisReasoner:
    """``HypothesisReasoner`` implementation backed by ``anthropic.Anthropic``.

    ``client`` may be any object exposing ``messages.create(...)`` (a fake in tests).
    """

    def __init__(
        self,
        client=None,
        *,
        model: str = "claude-opus-5",
        max_new: int = 3,
        max_confidence: float = 0.7,
        max_tokens: int = 4000,
        effort: str = "medium",
    ):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError('Install automl-py[llm] (pip install "anthropic") to use AnthropicHypothesisReasoner') from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_new = max_new
        self.max_confidence = max_confidence
        self.max_tokens = max_tokens
        self.effort = effort
        self.last_response_text: str | None = None

    # ------------------------------------------------------------------------------------------ #
    @staticmethod
    def facts(diagnosis: Diagnosis) -> list[tuple[str, str]]:
        """Numbered, verifiable statements derived from the diagnosis."""
        out: list[str] = []
        out.append(f"target={diagnosis.target}; metric={diagnosis.metric}; context={diagnosis.context or 'n/a'}")
        out.append("current feature columns: " + ", ".join(map(str, diagnosis.current_columns)))
        if diagnosis.baseline_scores:
            out.append(f"baseline fold scores: {[round(float(s), 4) for s in diagnosis.baseline_scores]}")
        rf = diagnosis.residual_frame
        if rf is not None and not rf.empty and "residual" in rf:
            res = rf.residual
            out.append(
                f"residual mean={float(res.mean()):.4f}, std={float(res.std()):.4f}, mean abs={float(res.abs().mean()):.4f}, n={len(res)}"
            )
            if diagnosis.timestamps is not None and len(diagnosis.timestamps) == len(rf):
                ts = np.asarray(diagnosis.timestamps)
                order = np.argsort(ts)
                q = np.array_split(order, min(4, len(order)))
                parts = [
                    f"period-quartile {i + 1}: mean abs residual {float(res.iloc[idx].abs().mean()):.4f}"
                    for i, idx in enumerate(q)
                    if len(idx)
                ]
                out.append("; ".join(parts))
            if diagnosis.groups is not None and len(diagnosis.groups) == len(rf):
                by = res.abs().groupby(np.asarray(diagnosis.groups)).mean().sort_values(ascending=False).head(3)
                out.append("segments with largest mean abs residual: " + ", ".join(f"{k}={v:.4f}" for k, v in by.items()))
        rd = diagnosis.residual_diagnostics
        if rd is not None and not rd.empty:
            top = rd.head(3)
            out.append(
                "residual associations: "
                + "; ".join(
                    f"{r.feature}: MI={float(r.residual_mutual_information):.3f}, corr={float(r.residual_correlation):+.3f}"
                    for r in top.itertuples()
                )
            )
        ms = diagnosis.missingness
        if ms is not None and not ms.empty and "missingness_predictability_auc" in ms:
            sys_ = ms[ms.missingness_predictability_auc.fillna(0) >= 0.65]
            if not sys_.empty:
                out.append(
                    "systematic missingness: "
                    + ", ".join(f"{r.feature} (AUC {r.missingness_predictability_auc:.2f})" for r in sys_.head(3).itertuples())
                )
        if diagnosis.drift and "auc" in diagnosis.drift:
            out.append(f"adversarial validation AUC={float(diagnosis.drift['auc']):.3f}")
        return [(f"F{i + 1}", text) for i, text in enumerate(out)]

    def _prompt(self, hypotheses: Sequence[Hypothesis], facts: list[tuple[str, str]]) -> str:
        payload = {
            "FACTS": [{"id": i, "text": t} for i, t in facts],
            "existing_hypotheses": [
                {
                    "id": h.hypothesis_id,
                    "statement": h.statement,
                    "kind": h.kind,
                    "concept": h.concept,
                    "confidence": round(h.confidence, 2),
                    "evidence": h.evidence,
                }
                for h in hypotheses
            ],
        }
        return json.dumps(payload, indent=1, default=str)

    def _call(self, system: str, user: str) -> str | None:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": user}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            log.warning("hypothesis reasoner: model refused; hypotheses left unchanged")
            return None
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                return block.text
        return None

    @staticmethod
    def _parse(text: str) -> dict | None:
        text = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
        if fenced:
            text = fenced.group(1)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------------------------------ #
    def refine(self, hypotheses: list[Hypothesis], diagnosis: Diagnosis) -> list[Hypothesis]:
        facts = self.facts(diagnosis)
        fact_text = dict(facts)
        try:
            text = self._call(SYSTEM_PROMPT.format(max_new=self.max_new), self._prompt(hypotheses, facts))
        except Exception as exc:
            log.warning("hypothesis reasoner failed (%s: %s); hypotheses left unchanged", type(exc).__name__, exc)
            return list(hypotheses)
        self.last_response_text = text
        data = self._parse(text) if text else None
        if not data:
            return list(hypotheses)

        out = list(hypotheses)
        by_id = {h.hypothesis_id: h for h in hypotheses}
        rerank = [i for i in data.get("rerank", []) if isinstance(i, str) and i in by_id]
        if rerank:
            ranked = [by_id[i] for i in dict.fromkeys(rerank)]
            out = ranked + [h for h in hypotheses if h.hypothesis_id not in set(rerank)]

        added = 0
        for item in data.get("new", []) or []:
            if added >= self.max_new or not isinstance(item, dict):
                continue
            ids = [i for i in item.get("evidence_ids", []) if isinstance(i, str) and i in fact_text]
            statement = str(item.get("statement", "")).strip()
            concepts = [str(c).strip() for c in item.get("search_concepts", []) if str(c).strip()]
            kind = str(item.get("kind", "external"))
            if not ids or not statement or not concepts or kind not in ALLOWED_KINDS:
                log.info("hypothesis reasoner: rejected proposal without cited evidence/concepts: %r", statement[:80])
                continue
            try:
                conf = float(item.get("confidence", 0.4))
            except (TypeError, ValueError):
                conf = 0.4
            added += 1
            out.append(
                Hypothesis(
                    f"H-LLM-{added:02d}",
                    statement,
                    [f"{i}: {fact_text[i]}" for i in ids],
                    concepts,
                    max(0.05, min(self.max_confidence, conf)),
                    kind=kind,
                    concept=f"llm:{concepts[0].lower()}",
                    signals={"source": "llm", "model": self.model},
                )
            )
        return out
