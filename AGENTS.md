# Repository Guidelines

This repository is in an early bootstrap state: no source code or commit history exists yet. The conventions below apply as the project is built out.

## Project Structure & Module Organization

- Source code lives in the repository root or a top-level package directory (e.g., `src/` or the package name), not in nested ad-hoc folders.
- Tests live in `tests/` and mirror the source layout.
- Configuration and assets (e.g., `.aws`, `.agents`, `.codex`) live in dot-directories at the root.
- Keep module responsibilities focused: one module, one concern. Import through public interfaces rather than internal helpers.

## Build, Test, and Development Commands

No build tooling exists yet. When it is added, document the canonical commands here, for example:

- `make build` — compile or package the project.
- `make test` / `pytest` / `npm test` — run the test suite.
- `make lint` / `npm run lint` — run formatting and lint checks.

Update this section whenever a command becomes the canonical way to build, test, or run the project.

## Coding Style & Naming Conventions

- Use 4-space indentation (Python) or the project's configured formatter defaults (e.g., Prettier for JavaScript).
- Follow language-idiomatic naming: `snake_case` for Python functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, `kebab-case` for web project filenames.
- Prefer descriptive names over abbreviations; avoid single-letter variables except for loop indices.
- Add a formatter and linter early (e.g., Black/Ruff for Python, ESLint/Prettier for TypeScript) and enforce them in CI.

## Testing Guidelines

- Choose a framework suited to the language (e.g., pytest for Python, Jest/Vitest for TypeScript).
- Name test files `test_<module>.py` or `<module>.test.ts` and name tests to describe behavior (e.g., `test_get_user_returns_404_for_missing_user`).
- Aim for meaningful coverage on critical paths; do not gate changes on arbitrary coverage percentages.

## Commit & Pull Request Guidelines

- Write imperative, concise commit messages (`feat: add user endpoint`, `fix: handle empty input`). Use conventional prefixes (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`) once commit history is established.
- Keep commits atomic: one logical change per commit.
- Open pull requests with a clear description of the change, motivation, and how it was tested. Link related issues and include screenshots for UI changes.
- Keep PRs small and reviewable; request review early.

## Agent-Specific Instructions

- `.codex/` and `.agents/` contain agent configuration. Treat their contents as local project instructions and respect any scoped AGENTS.md files as they appear.
