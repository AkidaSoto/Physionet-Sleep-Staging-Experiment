# Working in this repo

## Long-running commands

Keep command output out of the context window without hiding failures.

- **Training, build and experiment scripts** (`scripts/run_*.py`, `scripts/build_*.py`, `scripts/download_*.py`): redirect stdout and stderr to `logs/<script-name>.log` (`*.log` is gitignored) and run in the background when the job is long. Then read the last ~40 lines and grep the log for `Traceback|Error|nan`. Open the full log only when something failed or looks off.
- **Always check the exit code.** `cmd | tail` reports `tail`'s exit code, not `cmd`'s. Redirect to a file instead, or print `$?` / `${PIPESTATUS[0]}`. Never call a run successful on a truncated view alone.
- **Tests:** `pytest` runs quietly by default (`addopts = "-q"` in `pyproject.toml`); failures still print in full. On failure, rerun only the failing test with `-x`.
- **Dashboard build** (`showcase/`): show the tail of `npm run build`, along with its exit code.

## Code changes

- Build only what was asked. No unrequested features, config options or flags, and no abstractions for code used in one place.
- Make surgical edits. Don't refactor, rename or reformat code next to the change.
- Reuse existing helpers in `src/physionet_sleep/` and the standard library before adding a module or dependency.
- When a request is ambiguous, state the assumption instead of picking silently.
- **Never simplify away evaluation correctness:** subject-wise splits, leakage guards, held-out separation, metric computation and input validation stay, even when shorter code would "work".
