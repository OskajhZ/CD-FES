---
name: comment-cleanup
description: Strip trivial/inline comments from Python source files while preserving meaningful documentation
source: auto-skill
extracted_at: '2026-07-16T14:34:18.878Z'
---

# Python Comment Cleanup

Systematically remove trivial "what" comments from Python files while preserving meaningful "why" documentation.

## When to use

When a user asks you to clean up comments in code (e.g., "删掉无关注释", "清理注释", "remove unnecessary comments") before public release, or when preparing code for publication/archival.

## Procedure

### Step 1: Read all files first

Read every Python file in the target directory tree. Do **not** make edits until you've seen the full picture — comments that look trivial in isolation may reference each other.

### Step 2: Classify comments

Use this decision tree for every comment:

```
Is it a top-level file docstring (author, license, paper reference, pipeline overview)?
  → KEEP

Is it a function/class docstring explaining parameters, return values, or non-obvious behavior?
  → KEEP (but trim verbose docstrings to essential facts)

Is it an algorithm step-by-step that explains WHY a particular processing order or parameter choice was made?
  → KEEP (e.g., "4.5 Hz low-pass to avoid frontal alpha/theta being mistaken for EOG")

Is it a domain-knowledge annotation (channel mappings, electrode indices, paper citations)?
  → KEEP

Is it commented-out code that was disabled (not a config/documentation value)?
  → REMOVE

Is it an inline comment that merely restates what the code clearly does?
  (e.g., `# 带通滤波` above `raw.filter(...)`, `# 重建` above `ica.apply(...)`)
  → REMOVE

Is it a section divider like `# ===` or `# ---` that lines of code already make obvious?
  → REMOVE (or keep minimal structural headers if the file is very long)

Is it a "self-narrating" comment that describes obvious operations?
  (e.g., `# 从1/3处取`, `# 设置中文字体`, `# 标准化缩放`)
  → REMOVE
```

### Step 3: Edit systematically

- Use `edit` (targeted replacement) for surgical removals — this preserves author docstrings and meaningful comments.
- Use `write_file` (full rewrite) only when the file has many scattered trivial comments throughout, making targeted edits impractical.
- Process files in parallel where possible (multiple `edit` or `write_file` calls at once).

### Step 4: Preserve important structures

Keep **all** of the following intact:
1. Top-level file docstrings (author, year, paper title, purpose)
2. Academic references and citations
3. Algorithm rationale (why a specific parameter/order was chosen)
4. Channel mapping or domain-specific indices
5. Function/class docstrings that explain non-obvious semantics

## Example

Before:
```python
# Step 2: compute autocorrelation
K = np.zeros(...)
t = np.arange(0, length)
for tau in range(...):  # Use conjugate symmetry of K
    K[...] = ...  # compute at each time point
```

After:
```python
K = np.zeros(...)
t = np.arange(0, length)
for tau in range(...):
    K[...] = ...
```
(The `# Use conjugate symmetry of K` comment explains *why* this loop works → KEEP; the others are obvious.)

## Anti-patterns to avoid

- **Don't** merge files or change code structure — only remove comments.
- **Don't** touch meaningful docstrings that explain why a non-obvious approach was taken.
- **Don't** translate comments into another language unless explicitly asked.
- **Don't** remove comments that reference academic papers, even if they look noisy.
