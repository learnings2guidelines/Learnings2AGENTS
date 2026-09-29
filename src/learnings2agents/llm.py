"""Thin wrapper around the Gemini API (google-genai SDK) for LLM-mode synthesis.

Only imported/used when a Gemini API key is supplied; the tool works fully
without this module (see heuristics.py for the default, offline path).
"""

from __future__ import annotations

import json
import logging
import re

from learnings2agents.config import DEFAULT_GEMINI_MODEL, DEFAULT_GEMINI_MODEL_CHAIN
from learnings2agents.models import Learning, MergePlan, SynthesizedBullet, TextEdit

logger = logging.getLogger(__name__)


class LlmSynthesisError(RuntimeError):
    """Raised when the Gemini call fails or returns output we can't parse.

    Callers (synthesize.py) should catch this and fall back to heuristic mode
    for that directory rather than aborting the whole run.
    """


_SYSTEM_PROMPT = """\
You are helping build an AGENTS.md file section for one directory of a code \
repository, from raw "learnings" recorded by the CodeRabbit code review tool. \
Each numbered learning is a coding convention, review comment, or team \
decision that reviewers confirmed applies to files in this directory.

Task: merge duplicate or overlapping learnings, and rewrite the result as \
short, direct, imperative guidance suitable for an AGENTS.md file consumed by \
AI coding agents (e.g. "Use `TimeoutSampler` instead of bare `time.sleep()`"). \
Where several numbered learnings are specific instances of one general rule \
(e.g. several different "import X as a whole module, not `from X import y`" \
learnings for different modules), state the general rule once. Optionally \
assign a short thematic "heading" to a bullet (e.g. "Imports", "Testing", \
"Docstrings") when it helps organize a directory with several unrelated \
topics; leave it empty otherwise. Do not invent rules unsupported by the \
provided learnings. Drop learnings that are too narrow/one-off to generalize \
into useful guidance.

Respond with ONLY a JSON array (no surrounding prose, no markdown fences), \
where each element is an object:
  {"text": "<bullet text>", "heading": "<optional heading or empty string>", \
"sources": [<1-based numbers of the input learnings this bullet came from>]}
"""


def _build_prompt(directory: str, learnings: list[Learning]) -> str:
    lines = [
        f"Directory: {directory or '(repository root)'}",
        "",
        "Numbered learnings:",
    ]
    for i, learning in enumerate(learnings, start=1):
        lines.append(f"{i}. {learning.text}")
    lines.append("")
    lines.append("Produce the JSON array now.")
    return "\n".join(lines)


_MERGE_SYSTEM_PROMPT = """\
You are updating an existing AGENTS.md file for one directory of a code \
repository, given new "learnings" recorded by the CodeRabbit code review \
tool. You will be given the CURRENT FULL CONTENT of that directory's \
AGENTS.md file (it may contain hand-written prose, and/or a machine- \
generated block delimited by the exact marker lines \
`<!-- BEGIN CODERABBIT LEARNINGS -->` and `<!-- END CODERABBIT LEARNINGS -->`), \
plus a numbered list of new learnings, each tagged with its PR reference.

For each new learning:
- If the file ALREADY expresses an equivalent, overlapping, or more-specific- \
instance-of rule anywhere in the file (inside or outside the marker block), \
propose ONE edit that rewrites that existing snippet in place to incorporate \
the new learning, instead of creating a duplicate elsewhere. `old_text` MUST \
be an EXACT, VERBATIM substring copied character-for-character from the \
CURRENT FULL CONTENT above (do not paraphrase or reformat it) — keep it as \
short as possible, ideally just the one bullet/line being changed, so it can \
be located unambiguously. If the existing snippet already ends with a PR \
reference like `(PR #100)` or `(PRs #100, #101)`, extend it with the new \
learning's PR number(s) rather than dropping the existing one(s).
- If a learning does not match anything already in the file, add it to \
"new_bullets" instead (it will be appended into the marker block for you; do \
not try to place it in the file content yourself).
- Never propose an edit that deletes or shortens existing information \
without replacing it with equivalent-or-richer guidance.

Respond with ONLY a JSON object (no surrounding prose, no markdown fences) \
of the form:
  {"edits": [{"old_text": "<exact existing snippet>", \
"new_text": "<rewritten replacement>"}], \
"new_bullets": [{"text": "<bullet text>", \
"heading": "<optional heading or empty string>", \
"sources": [<1-based numbers of the input learnings this bullet came from>]}]}
Either list may be empty. Do not include both an edit and a new_bullets entry \
for the same learning.
"""


def _build_merge_prompt(
    directory: str, learnings: list[Learning], existing_content: str
) -> str:
    lines = [
        f"Directory: {directory or '(repository root)'}",
        "",
        "=== CURRENT FULL AGENTS.md CONTENT ===",
        existing_content,
        "=== END CURRENT FULL AGENTS.md CONTENT ===",
        "",
        "New numbered learnings:",
    ]
    for i, learning in enumerate(learnings, start=1):
        pr_note = (
            f"[PR #{learning.pull_request}]" if learning.pull_request else "[no PR]"
        )
        lines.append(f"{i}. {pr_note} {learning.text}")
    lines.append("")
    lines.append("Produce the JSON patch object now.")
    return "\n".join(lines)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(raw_text: str) -> str:
    """Strip ```json ... ``` fences if the model wrapped its output in them."""
    match = _FENCE_RE.search(raw_text)
    return match.group(1).strip() if match else raw_text.strip()


def _unique_pull_requests(learnings: list[Learning]) -> list[str]:
    seen: list[str] = []
    for learning in learnings:
        if learning.pull_request and learning.pull_request not in seen:
            seen.append(learning.pull_request)
    return seen


class GeminiClient:
    """Lazily-initialized wrapper around `google.genai.Client`."""

    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL):
        if not api_key:
            raise ValueError("GeminiClient requires a non-empty api_key.")
        self.api_key = api_key
        self.model = model
        self._client = None

    @classmethod
    def from_model_chain(
        cls,
        api_key: str,
        models: list[str] = DEFAULT_GEMINI_MODEL_CHAIN,
    ) -> GeminiClient:
        """Try each model in *models* with a lightweight probe and return the
        first one that responds successfully.

        Raises ``LlmSynthesisError`` if every candidate fails (e.g. all are
        overloaded or the key is invalid).
        """
        errors: list[str] = []
        for model in models:
            candidate = cls(api_key=api_key, model=model)
            try:
                client = candidate._get_client()
                client.models.generate_content(
                    model=model,
                    contents="Say OK",
                    config={"temperature": 0.0, "max_output_tokens": 4},
                )
                logger.info("Model probe succeeded: using %s.", model)
                return candidate
            except Exception as exc:  # noqa: BLE001
                logger.warning("Model %s unavailable (%s), trying next…", model, exc)
                errors.append(f"{model}: {exc}")

        raise LlmSynthesisError(
            "No Gemini model from the fallback chain responded successfully.\n"
            + "\n".join(errors)
        )

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - exercised only w/o dep
                raise LlmSynthesisError(
                    "google-genai is not installed. Install it (see requirements.txt) "
                    "or omit --gemini-api-key to use heuristic mode."
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def synthesize_directory(
        self, directory: str, learnings: list[Learning]
    ) -> list[SynthesizedBullet]:
        """Call Gemini once for `directory` and return parsed `SynthesizedBullet`s.

        Raises `LlmSynthesisError` on any API failure or unparsable response;
        callers should catch it and fall back to heuristic mode.
        """
        client = self._get_client()
        prompt = _build_prompt(directory, learnings)

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as exc:  # network/auth/quota errors, etc.
            raise LlmSynthesisError(
                f"Gemini API call failed for directory '{directory or '(root)'}': {exc}"
            ) from exc

        raw_text = getattr(response, "text", None) or ""
        try:
            payload = json.loads(_extract_json(raw_text))
        except json.JSONDecodeError as exc:
            raise LlmSynthesisError(
                f"Could not parse Gemini JSON output for directory "
                f"'{directory or '(root)'}': {exc}. Raw output: {raw_text[:500]!r}"
            ) from exc

        if not isinstance(payload, list):
            raise LlmSynthesisError(
                f"Gemini output for directory '{directory or '(root)'}' was not a "
                f"JSON array: {raw_text[:500]!r}"
            )

        all_prs = _unique_pull_requests(learnings)
        bullets: list[SynthesizedBullet] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            heading = str(item.get("heading") or "").strip()

            source_indices = item.get("sources") or []
            prs: list[str] = []
            for idx in source_indices:
                try:
                    learning = learnings[int(idx) - 1]
                except (TypeError, ValueError, IndexError):
                    continue
                if learning.pull_request and learning.pull_request not in prs:
                    prs.append(learning.pull_request)
            if not prs:
                # Model omitted/garbled sources; fall back to attributing every
                # PR in the directory rather than losing attribution entirely.
                prs = all_prs

            bullets.append(
                SynthesizedBullet(text=text, pull_requests=prs, heading=heading)
            )

        if not bullets:
            raise LlmSynthesisError(
                f"Gemini returned no usable bullets for directory '{directory or '(root)'}'."
            )
        return bullets

    def plan_agents_md_merge(
        self, directory: str, learnings: list[Learning], existing_content: str
    ) -> MergePlan:
        """Ask Gemini for a small patch that folds `learnings` into an
        existing AGENTS.md file's `existing_content`, instead of regenerating
        the whole file.

        Returns a `MergePlan` (possibly with empty `edits`/`new_bullets` if
        the model found nothing to change — that's a valid outcome, not an
        error). Raises `LlmSynthesisError` on any API failure or unparsable/
        malformed top-level response; callers should catch it and fall back
        to the marker-only bullets-based synthesis path for that directory.
        """
        client = self._get_client()
        prompt = _build_merge_prompt(directory, learnings, existing_content)

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "system_instruction": _MERGE_SYSTEM_PROMPT,
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as exc:  # network/auth/quota errors, etc.
            raise LlmSynthesisError(
                f"Gemini merge-plan call failed for directory "
                f"'{directory or '(root)'}': {exc}"
            ) from exc

        raw_text = getattr(response, "text", None) or ""
        try:
            payload = json.loads(_extract_json(raw_text))
        except json.JSONDecodeError as exc:
            raise LlmSynthesisError(
                f"Could not parse Gemini merge-plan JSON output for directory "
                f"'{directory or '(root)'}': {exc}. Raw output: {raw_text[:500]!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise LlmSynthesisError(
                f"Gemini merge-plan output for directory '{directory or '(root)'}' "
                f"was not a JSON object: {raw_text[:500]!r}"
            )

        edits: list[TextEdit] = []
        for item in payload.get("edits") or []:
            if not isinstance(item, dict):
                continue
            old_text = str(item.get("old_text", ""))
            new_text = str(item.get("new_text", ""))
            if not old_text or not new_text or old_text == new_text:
                logger.debug(
                    "Dropping malformed/no-op merge edit for directory '%s': %r",
                    directory or "(root)",
                    item,
                )
                continue
            edits.append(TextEdit(old_text=old_text, new_text=new_text))

        all_prs = _unique_pull_requests(learnings)
        new_bullets: list[SynthesizedBullet] = []
        for item in payload.get("new_bullets") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            heading = str(item.get("heading") or "").strip()

            source_indices = item.get("sources") or []
            prs: list[str] = []
            for idx in source_indices:
                try:
                    learning = learnings[int(idx) - 1]
                except (TypeError, ValueError, IndexError):
                    continue
                if learning.pull_request and learning.pull_request not in prs:
                    prs.append(learning.pull_request)
            if not prs:
                # Model omitted/garbled sources; fall back to attributing every
                # PR in this run rather than losing attribution entirely (same
                # fallback as synthesize_directory — these are always
                # brand-new bullets with no existing counterpart to inherit
                # attribution from).
                prs = all_prs

            new_bullets.append(
                SynthesizedBullet(text=text, pull_requests=prs, heading=heading)
            )

        return MergePlan(edits=edits, new_bullets=new_bullets)
