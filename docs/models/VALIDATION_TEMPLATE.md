# Validation template

Copy this block into `VALIDATION.md` for each model. The three middle
headings are SR 11-7's three core elements of validation (section V):
conceptual soundness, ongoing monitoring and outcomes analysis.

```markdown
### `ID`: model name

Tier N, kind, status `...`, validated YYYY-MM-DD, next review YYYY-MM-DD.

**Purpose.** What decision or report uses the output.

**Conceptual soundness.** The model in one or two sentences, the
assumptions it rests on, and the reference it follows (paper, standard or
book, with section). Say why the assumptions are acceptable for this use.

**Outcomes analysis.** The tests that compare outputs with something
known: closed forms, published tables, other libraries, the other language
engines, backtests. Cite each as `path::test`. Give the tolerance and the
worst observed deviation where the test reports one.

**Ongoing monitoring.** The KPIs from the inventory, where they are
computed, and what value would trigger a review.

**Limitations and compensating controls.** Where the model is known to be
wrong, and what the code does about it (guards, warnings, refusals).

**Findings.** Open issues found during validation, each with an owner and
a due date, or "none".
```

Rules for filling it in:

- Every claim cites a test, a file or a reference. "Accurate" without a
  number and a test is not a validation statement.
- `independently-checked` needs at least one outcome test against code or
  data the model does not share.
- A finding that changes a number goes in the CHANGELOG with the before
  and after values.
