# Agent Rules

## Project Phase

This project may become a full software system later, but the current phase is focused on the computer vision foundation first.

For now, prioritize understanding and building the CV parts carefully before moving into product, backend, frontend, deployment, or large architecture work.

Do not rush into productization unless the user explicitly asks for it.

## 1. Think First, Code Later

Before writing code, explain what you are about to do, why that approach fits, and what alternatives exist.

Do not jump straight to implementation.

Start with a short brainstorming message.

Be verbose enough to make the reasoning clear. Think out loud during brainstorming, planning, trade-off analysis, and implementation updates so the user can follow the decision process.

Wait for explicit approval such as:

```text
go ahead
implement it
start coding
```

before writing code.

## 2. Small Steps Only

Break features into the smallest logical, testable unit.

One step should correspond to one focused change.

Do not bundle multiple unrelated features into one edit.

If a change feels large, split it and ask which part to start with.

## 3. Always Explain Every Change

For every file or function you add, explain what it does and why it exists.

If you add a dependency, explain what it is, why it fits, and whether a simpler option exists.

## 4. Ask Before Assuming

If behavior, UI, shortcuts, library choice, or architecture is unclear, ask.

If you must choose a default, say what default you picked and why.

## 5. Never Delete Or Overwrite Silently

If you need to replace working code, show the before/after intent and explain the reason.

Ask for approval before replacing behavior that already works.

## 6. Smallest-Part Commit Workflow

Commit each logical, testable part separately.

A commit should represent one focused change that can be reviewed and verified on its own.

Do not bundle unrelated edits into the same commit.

## 7. Commit Message Guidelines

Use Conventional Commits in this short format:

```text
<type>[optional scope]: <description>
```

Use `feat:` for new features and `fix:` for bug fixes.

Use `docs:`, `test:`, `refactor:`, `style:`, `chore:`, `build:`, or `ci:` when they fit.

Add a scope when it clarifies the area, such as:

```text
feat(ocr): add text extraction
```

Mark breaking changes with `!` before the colon or a `BREAKING CHANGE:` footer.

Examples:

```text
docs: add commit guidelines
fix(ocr): handle empty output
feat(ui): add capture overlay
```
