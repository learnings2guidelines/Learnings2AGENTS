# learnings2agents

A tool to turn a CodeRabbit **Learnings** CSV export into scoped `AGENTS.md`
files, written directly into the folders of a target repository checkout.

Learnings are grouped by the directory of the file they were recorded
against (the CSV's `File` column) and one `AGENTS.md` is generated per
directory from the start — there is no single monolithic file that gets
split up afterwards. See [`## Design notes`](#design-notes) below for why.

The tool works fully offline out of the box (no API key required). If you
provide a Gemini API key, per-directory synthesis is upgraded to use the LLM
for higher-quality deduplication and rewriting — see
[`## What does a Gemini API key add?`](#what-does-a-gemini-api-key-add).

## Quick start

```bash
git clone <this repo> && cd Learnings2AGENTS
make run CSV=/home/rkishner/Downloads/learnings.csv TARGET=/home/rkishner/github_repo/hyperconverged-cluster-operator
```

That's it — `make run` creates a local virtualenv, installs dependencies,
and writes/merges `AGENTS.md` files into `TARGET`, mirroring the folder
structure referenced by the CSV.

Optional extras:

```bash
make run CSV=learnings.csv TARGET=/path/to/repo \
  REPOSITORY=my-org/my-repo \  # only needed if the CSV has multiple repos
  MODEL=gemini-2.5-flash \     # optional: override the Gemini model (default: gemini-2.5-flash)
  SHOW_LIB_LOGS=1              # optional: also show INFO logs from third-party
                                # libs (httpx, google-genai), e.g. the Gemini
                                # HTTP request lines. Off by default to reduce
                                # noise; equivalent to `--show-lib-logs`.
```

To enable LLM-mode synthesis, **export** `GEMINI_API_KEY` rather than passing
it as a `make` variable on the command line:

```bash
export GEMINI_API_KEY=xxxx
make run CSV=learnings.csv TARGET=/path/to/repo
```

See [`## Handling the Gemini API key safely`](#handling-the-gemini-api-key-safely)
for why, and how to wire this up in GitHub Actions.

Preview what would be written without touching disk:

```bash
make dry-run CSV=learnings.csv TARGET=/path/to/repo
```

Run the test suite:

```bash
make test
```

You can also invoke the CLI directly for the full set of flags:

```bash
python -m learnings2agents.cli --help
```

## What it does

1. Parses the CodeRabbit CSV export (`Learning, Repository, File, Pull Request,
   URL, Created By, Usage, Last Used, Created At, Updated At`).
2. Selects a single repository's rows (auto-detected if the CSV only has
   one; pass `--repository`/`REPOSITORY=` to disambiguate otherwise).
3. Groups learnings by the directory of their `File` column and verifies
   that directory exists under `--target`/`TARGET`.
4. For each directory, synthesizes a deduplicated, imperative-style bullet
   list — either via the Gemini API (if a key is given) or a fully offline
   heuristic (near-duplicate text clustering via `difflib`).
5. Writes/merges an `AGENTS.md` per directory. **Creating** a directory's
   file from scratch (it didn't exist before this run) writes plain
   generated content with no extra markers — it's 100% generated, so
   there's nothing to distinguish yet. **Updating** an already-existing
   file wraps the new content in `<!-- BEGIN/END CODERABBIT LEARNINGS -->`
   markers instead, so any hand-written content in that file is preserved
   and a *future* run can find/replace just the generated section. This is
   decided independently per directory — one directory being created fresh
   doesn't affect whether a sibling directory's pre-existing file gets
   markers. If a Gemini key is given **and** that directory's `AGENTS.md`
   already exists, step 4 is skipped in favor of a patch-based merge into
   the existing file instead
   — see [`## What does a Gemini API key add?`](#what-does-a-gemini-api-key-add).

Results are cached (`<target>/.learnings2agents_cache/` by default, disable
with `--no-cache`) so re-running against an unchanged CSV/target skips
redundant LLM calls.

## What does a Gemini API key add?

Without a key, the tool is fully functional: it groups learnings by folder,
deduplicates near-identical text with `difflib`, and writes an `AGENTS.md`
per folder. A Gemini key upgrades the quality of that per-directory
synthesis step:

- **Semantic (not just textual) deduplication** — merges rows that express
  the same rule in different words, which plain text-similarity would miss.
- **Rewriting into clean, imperative guidance** — turns a review-comment
  style sentence into a short, direct AGENTS.md bullet.
- **Thematic grouping within a folder** — related bullets get grouped under
  short subheadings (e.g. "Imports", "Testing") instead of one flat list.
- **Generalizing repeated specific instances into one broader rule** within
  a folder (e.g. several "import X as a whole module" learnings for
  different modules become one general bullet).
- **Noise filtering** of very narrow, one-off learnings.
- **Patch-based merging into an already-existing `AGENTS.md`** — if a
  directory's `AGENTS.md` already exists (e.g. from a previous run, or
  hand-written), Gemini is shown the *entire* file as context, but only
  returns a small patch: exact-text edits for existing rules — anywhere in
  the file, hand-written or previously generated — that a new learning
  restates or specializes, plus brand-new bullets (added inside the
  `<!-- BEGIN/END CODERABBIT LEARNINGS -->` block) for learnings that match
  nothing existing. This keeps token cost/latency proportional to what
  actually changed rather than the size of the whole file, and an edit that
  can't be matched exactly in the file is safely skipped (logged as a
  warning) rather than corrupting it. If the merge-planning call itself
  fails (network/parsing error), that directory automatically falls back to
  the regular marker-section-only synthesis below instead of aborting the
  run.
- **Consolidating a brand-new repository into one root `AGENTS.md`** — if
  the target had **no** `AGENTS.md` files at all before this run, Gemini is
  given *every* directory's learnings at once (each tagged with its
  original directory) and asked to make a best effort to fold them into a
  single, thematically-organized root-level `AGENTS.md`, instead of writing
  one small file per directory. This is a "try", not a guarantee: if the
  consolidation call fails or returns nothing usable, the tool transparently
  falls back to the regular one-file-per-directory pipeline described
  above. See [`## Design notes`](#design-notes) for the trade-off this
  addresses.
- **A final redundancy re-check** — right after a directory's (or, in
  consolidated-root mode, the whole run's) new bullets are finalized,
  Gemini is asked once more whether any of those *new* bullets are already
  covered by something else — either existing content already in the file,
  or another new bullet earlier in the same batch — and any that are get
  dropped before writing, rather than leaving a near-duplicate rule behind.
  A failure here is logged and simply keeps all the new bullets as-is; it
  never blocks a write.

None of this is required to use the tool; it only affects how polished the
generated bullets are. The model used defaults to `gemini-2.5-flash`
(`DEFAULT_GEMINI_MODEL` in `config.py`) unless overridden by the
`GEMINI_MODEL` env var or the `--model`/`MODEL=` flag.

## Handling the Gemini API key safely

`GEMINI_API_KEY` is optional and is only ever read from the **environment**
(`--gemini-api-key` also exists as an explicit CLI flag if you need it, e.g.
for scripting, but prefer the environment variable). Concretely:

- The `Makefile`'s `run`/`dry-run` recipes never reference
  `$(GEMINI_API_KEY)` and are silenced (`@`), so no command line containing a
  secret is ever echoed to your terminal or CI logs. GNU Make automatically
  forwards environment/command-line variables into recipe subshells, so the
  key still reaches the Python process via `os.environ` without appearing in
  any printed command.
- Locally: `export GEMINI_API_KEY=...` before `make run`.
- In GitHub Actions: put the key in a **GitHub Environment** secret
  (Settings → Environments → your environment → Secrets), then reference it
  from a job that targets that environment:

  ```yaml
  jobs:
    generate:
      runs-on: ubuntu-latest
      environment: production   # <- makes secrets.GEMINI_API_KEY resolve to
                                 #    that Environment's secret
      steps:
        - uses: actions/checkout@v4
        - name: Generate AGENTS.md
          env:
            GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          run: make run CSV=learnings.csv TARGET=./target-repo
  ```

  A ready-to-customize version of this lives at
  [`.github/workflows/generate-agents-md.yml`](.github/workflows/generate-agents-md.yml)
  (manual `workflow_dispatch` trigger only).
- `tests/test_security.py` regression-tests that the key never shows up in
  logs, stdout/stderr, `--help` output, or the `Makefile` text itself.

### Running the API-key/model test workflows on pull requests from forks

`.github/workflows/{api-key-model,api-key-only,model-only}-tests.yml` need
`GEMINI_API_KEY`/`GEMINI_MODEL` to run, but GitHub never passes repository
secrets or variables to a `pull_request`-triggered run whose head branch is
a fork — regardless of any repo setting. Each of these workflows therefore
uses a single job triggered only by `push` and `pull_request_target`
(`pull_request_target` covers *all* pull requests, fork or not, so there's
no separate `pull_request` trigger and the workflow doesn't fire twice per
PR update). The job's `environment:` is computed by an expression:

- For `push` and same-repo pull requests, it evaluates to an empty string —
  GitHub treats that as "no environment", so secrets/variables resolve
  straight from the repo-level ones, no approval step, same as before.
- For a `pull_request_target` run whose head repo is an actual fork, it
  evaluates to `gemini-ci-fork` instead — a dedicated **Environment** that
  gates the job behind a required reviewer before its secret/variable are
  resolved.

`pull_request_target` runs with the base repository's trust level for *any*
fork, on first contact, with no approval gate by default — checking out and
running a fork's own code in that context (needed here, since that's the
code under test) would hand the real key to unreviewed code, which could
exfiltrate it (e.g. a network call from within the test suite). GitHub's
automatic log secret-masking does **not** protect against this: it only
scrubs literal matches from log text, so it neither catches an outbound
network request nor a trivially obfuscated `print` (base64, reversed,
char-split, ...). It also always uses the workflow file from the base
branch (not the PR's) and defaults to checking out the base branch too —
each workflow sets `ref:` explicitly on the checkout step so it actually
tests the PR's code once approved.

To close the secrets gap for fork PRs, configure the `gemini-ci-fork`
Environment once per repository with:

1. **Settings → Environments → New environment**, named `gemini-ci-fork`.
2. **Required reviewers** enabled on it, with at least one maintainer added
   — this pauses the job right before the environment's secret/variable are
   resolved, so a human explicitly reviews the diff and approves each fork
   PR run before its code gets access to the real key.
3. An environment secret `GEMINI_API_KEY` and environment variable
   `GEMINI_MODEL`, scoped to `gemini-ci-fork` only. Use a **separate,
   low-quota, CI-only Gemini key** here (not the one used for `push`/
   same-repo runs or for `generate-agents-md.yml`), with usage/billing
   alerts, and rotate it periodically — a reviewer approving a run is a
   strong mitigation, not a guarantee, against a well-hidden malicious diff.

## Design notes

The tool builds **several scoped `AGENTS.md` files from the start** rather
than generating one big file and splitting it afterwards: each directory's
`AGENTS.md` is synthesized independently from only the learnings whose
`File` falls under that directory. This maps directly onto the CSV's
existing per-file structure, is cheap to cache/re-run, and scales to CSVs
covering many repositories. The trade-off is that if the exact same
convention was learned separately in several different directories, it will
appear once (deduplicated) in each of those directories' files rather than
being consolidated into one parent/root file. The one exception is a
brand-new target (no `AGENTS.md` files anywhere yet) with a Gemini key
available: there, the tool instead makes a best effort to consolidate
*everything* into a single root `AGENTS.md` from the start (see
[`## What does a Gemini API key add?`](#what-does-a-gemini-api-key-add)) —
once that root file (or any `AGENTS.md`) exists, later runs go back to the
regular per-directory split described above.

**A note on merging into existing files:** when a Gemini key is used and a
directory's `AGENTS.md` already exists, the LLM is given the whole file as
context and can propose an edit to *any* matching rule in it, including
hand-written prose outside the `<!-- BEGIN/END CODERABBIT LEARNINGS -->`
block. Each proposed edit is only applied if its exact quoted snippet is
found once, unambiguously, in the current file — a paraphrased or missing
snippet is skipped (logged, not applied) rather than guessed at — but as
with any LLM-assisted rewrite of freeform text, it's worth reviewing the
diff on the first run against a heavily hand-curated `AGENTS.md`.

## Repository layout

```
.github/workflows/
  generate-agents-md.yml  # example workflow_dispatch CI job (see above)
src/learnings2agents/
  cli.py           # argparse entrypoint, orchestrates the pipeline
  config.py         # defaults: model name, similarity threshold, cache dir
  models.py         # Learning / DirGroup / SynthesizedBullet / TextEdit / MergePlan
  csv_loader.py      # parse the CSV, filter/auto-detect repository
  tree.py            # group learnings by directory, verify against target;
                       # target_has_any_agents_md() detects a fresh target
  llm.py             # Gemini client wrapper (google-genai): fresh-file synthesis
                       # (synthesize_directory), existing-file merge planning
                       # (plan_agents_md_merge), cross-directory root
                       # consolidation (synthesize_root), and the final
                       # redundancy re-check (dedupe_bullets) — all used only
                       # with a key
  heuristics.py       # offline fallback: difflib-based near-duplicate clustering
  synthesize.py        # per-directory synthesis orchestration + cache lookup;
                        # routes to the merge-plan path when a directory's
                        # AGENTS.md already exists (LLM mode only); also runs
                        # the dedupe_bullets re-check before finalizing bullets.
                        # synthesize_consolidated_root() is the single-root-file
                        # alternative used for a fresh target + a Gemini key.
  writer.py            # render Markdown, merge into existing AGENTS.md files;
                        # apply_merge_plan() applies an LLM merge plan's edits +
                        # new bullets onto an existing file's full content;
                        # write_agents_md() omits the CODERABBIT LEARNINGS
                        # markers when creating a directory's file from
                        # scratch, and includes them when updating one that
                        # already existed
  cache.py             # on-disk content-hash cache for synthesis results and
                        # merge plans (the latter also keyed on file content)
tests/
  fixtures/sample_learnings.csv
  test_security.py     # regression tests: API key is never printed/logged
  test_*.py
```
