# Contributing to Slipstream

Thank you for helping improve Slipstream. Contributions from people and from
AI agents are both welcome, and so is code written with AI assistance. The
same standard applies to every change: it must be correct, tested,
documented and understood by the person who submits it.

This file is the single source of repository instructions. `AGENTS.md` and
`CLAUDE.md` point agents here; personal agent settings such as `.claude/` and
`.agents/` are git-ignored and stay local.

## Ways to contribute

- **Report a bug.** Open an issue with the smallest script that reproduces it,
  the expected and actual result, and your versions: Slipstream, Python,
  PyTorch, operating system, and device (CPU or GPU model).
- **Suggest a feature or physics model.** Open an issue first for anything
  larger than a small fix, so the design can be agreed before you invest time.
  Cite the reference or code you want to match.
- **Improve documentation, examples or tests.** These are always useful and
  need no prior discussion.
- **Fix an issue.** Comment on the issue so work is not duplicated, then open a
  pull request.

Security problems: do not post details in a public issue. Use the repository's
**Security → Report a vulnerability** form on GitHub.

## Development setup

Python 3.10 or newer is required. A GPU is not required: CI and most tests run
on CPU.

```bash
git clone https://github.com/<your-username>/slipstream.git
cd slipstream
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# CPU-only PyTorch, then the package in editable mode with test and lint tools.
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple torch
python -m pip install -e ".[sdds]" pytest ruff
```

For the CUDA and Warp backend, use `requirements.txt` instead. See
[getting started](docs/getting_started.md) and
[compilation and CUDA](docs/compilation.md) for toolchains, including the
Windows `clang-cl` requirement for `torch.compile`.

## Workflow

1. Fork the repository and create a branch from an up-to-date `main`. Name it
   by purpose: `feature/<topic>`, `fix/<topic>`, `docs/<topic>` or
   `chore/<topic>`.
2. Make one focused change per branch. Unrelated refactors and formatting
   churn belong in separate pull requests.
3. Run the checks below until they pass.
4. Push and open a pull request against `main`. Link the issue it closes.
5. Respond to review. Push follow-up commits rather than force-pushing over
   review history; the maintainers merge with a merge commit.

Commit messages use a short imperative summary, for example
`Add per-kind kick counts to RingMap`. Add a body when the reason for the
change is not obvious from the diff.

## Checks before you push

Every push runs the fast physics tests and the linter. Pull requests and manual
runs also run the full suite on Python 3.10, 3.12 and 3.13. Run them locally first:

```bash
python -m pytest -m "physics and not slow"   # fast physics tests, while developing
python -m pytest                             # full test suite, before you push
ruff check src tests                         # lint
```

Optionally confirm the package still builds:

```bash
python -m pip install build twine
python -m build && python -m twine check dist/*
```

A pull request is merged only when CI is green.

## Code standards

- **Style.** Ruff enforces the rules in `pyproject.toml` (line length 120).
  Match the style of the surrounding code; do not reformat files you are not
  otherwise changing.
- **Scope.** Keep the diff as small as the change allows. Do not add
  abstractions, options or dependencies for hypothetical future needs.
- **Dependencies.** The core install stays light: `numpy`, `scipy`, `pandas`,
  `h5py`, `PyYAML` and `torch`. Anything heavier goes behind an optional extra
  in `pyproject.toml`, and the code must work, or fail with a clear message,
  when the extra is missing.
- **CPU first.** Every feature must run on CPU. GPU and Warp paths are
  accelerations and must agree with the CPU result within a stated tolerance.
- **Conventions.** Particle states are canonical `(x, px, y, py, tau, delta)`
  with `tau` in metres and energies in eV including rest energy. See the
  [native physics reference](docs/native_reference.md) before touching
  transport, and state units in every new public argument.
- **Differentiability.** Code that sits in an autograd or CUDA-graph path must
  not break gradients or capture. Add a gradient test when you change such
  code.
- **Errors.** Validate inputs at public boundaries and raise `ValueError` or
  `TypeError` with a message naming the offending argument.

## Tests

- Every bug fix needs a test that fails without the fix. Every new behavior
  needs a test that exercises it.
- Tests live in `tests/` as `test_*.py` and use `pytest`. Keep them fast and
  deterministic: seed random numbers and avoid network access.
- Every test file has a tier, set in `tests/conftest.py`: `physics` (numbers
  checked against closed forms or references), `workflow` (a specific use such
  as LOCO or an example study) or `unit` (one function or interface). Add a new
  file to `TIERS` there. Mark tests that take more than a few seconds with
  `@pytest.mark.slow`. While developing, run the fast physics set:
  `python -m pytest -m "physics and not slow"` (about a minute).
- Tests that need CUDA must skip cleanly without it, as in
  `tests/test_collective_gradients.py`:
  `pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")`.
- For physics, test against an analytic limit, a conserved quantity or a
  reference code (elegant, ASTRA, Cheetah, Xsuite) rather than against the
  output of the code being changed.
- Do not commit generated results or large data. `examples/` tracks source
  files only; new captures need explicit justification in the pull request.

## Documentation

Behavior changes update the docs in the same pull request.

- `docs/modules/` has one page per module, split into **Physics** (model,
  units, approximations, limits) and **Usage** (inputs, outputs, examples).
  The index in `docs/modules/README.md` lists every page.
- Add or update a docstring for every public function and class. Use comments
  only for non-obvious local choices.
- Call an algorithm validated only after recording the model, inputs, dtype,
  device and reference used. See "Maintaining documentation" in
  [docs/README.md](docs/README.md).
- Keep current behavior in the docs. Do not leave plans or history there.

## Contributions written with AI

AI assistance is welcome here. The rules are the ones that keep review
tractable and the physics trustworthy.

**For the human submitting:**

- You are the author of record and are accountable for the change. Read and
  understand every line before you open the pull request.
- Say in the pull request description that AI helped, and which tool, if it
  wrote a substantial part. Add a `Co-Authored-By:` trailer to commits when
  your tool supports it.
- Run the checks above yourself. "The agent said it passes" is not evidence.

**For AI agents:**

- Read this file, `docs/README.md` and the module pages for the code you
  change before editing.
- Do not invent results. Never fabricate benchmark numbers, validation
  claims, reference papers, equations or citations. If you did not run it or
  verify it, say so.
- Do not weaken or delete a test, widen a tolerance or add a skip to make a
  check pass. If a test seems wrong, explain why in the pull request.
- Keep the change small and in scope. Do not rewrite unrelated files, rename
  public APIs or bump dependency versions unprompted.
- Do not commit secrets, local paths, caches or personal agent configuration
  (`.env`, `.claude/`, `.agents/`, memory files). The `.gitignore` covers
  most of these; check `git status` anyway. Edit `AGENTS.md` only to change
  repository-wide agent guidance, and keep it short.
- Do not copy code from other projects unless its licence is compatible with
  Apache-2.0 and the source is credited. Treat code from elegant, ASTRA and
  similar codes as references to match behavior, not text to paste.
- Report plainly what you ran and what failed. A partial result stated
  honestly is better than a complete result that is not.

**For reviewers:** AI-assisted pull requests get the same review as any
other. Large, unfocused or unverified submissions may be closed with a request
to split them or add evidence.

## Pull request checklist

Copy this into the pull request description.

```markdown
- [ ] One focused change; linked issue: #
- [ ] `python -m pytest` passes locally
- [ ] `ruff check src tests` passes locally
- [ ] New or changed behavior has tests (CUDA tests skip without a GPU)
- [ ] Docs and docstrings updated (`docs/modules/…` if a module changed)
- [ ] Units and coordinate conventions stated for new public arguments
- [ ] No generated results, secrets, caches or agent-local files committed
- [ ] AI assistance disclosed (tool, and which parts), if substantial
```

## Releasing

Releases are published to PyPI as `slipstream-accel` by
`.github/workflows/publish.yml` using
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API
token is stored. PyPI filenames are permanent: a version that has been uploaded
can never be uploaded again, even after deletion.

Version 1.1.0 was uploaded by hand, which created the PyPI project. Later
releases use the workflow.

One-time setup, by a maintainer of the PyPI project:

1. On pypi.org, open the `slipstream-accel` project, then **Manage → Publishing**,
   and add a GitHub trusted publisher: owner `astec-stfc`, repository
   `slipstream`, workflow `publish.yml`, environment `pypi`.
2. On GitHub, create the environment `pypi` under **Settings → Environments**.
   Optionally add yourself as a required reviewer so each upload needs approval.

For each release:

1. Set `version` in `pyproject.toml` and merge the change to `main`.
2. Publish a GitHub release whose tag is `v<version>` (for example `v1.1.1`).
   The workflow refuses to publish if the tag and `pyproject.toml` disagree.
3. To publish the current `main` without a new release, run **Actions →
   Publish to PyPI → Run workflow** on `main`. This fails if that version is
   already on PyPI, so bump `version` first.

After the upload, check that `pip install slipstream-accel==<version>` works in
a clean environment.

## Licence and conduct

Slipstream is released under the [Apache License 2.0](LICENSE). By submitting a
contribution you agree that it is licensed under the same terms, as set out in
section 5 of that licence. Only submit work you have the right to contribute.

Be respectful and constructive in issues, reviews and discussion. Assume good
faith, critique code and not people, and welcome newcomers. Maintainers may
edit or remove comments and contributions that are abusive or off-topic.
