# Technical report

The report is authored in [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) and exported
directly to [`notgpu-attention-technical-report.pdf`](notgpu-attention-technical-report.pdf).
There is no HTML source or HTML rendering step.

The report is intentionally concise and covers the required submission items:

- how the solution addresses the problem statement;
- development tools;
- APIs and AI development tools;
- libraries and frameworks; and
- datasets and project assets.

Five figures under [`figures/`](figures/) are screenshots exported from pages 5, 12,
14, 15, and 16 of the project presentation.

## Regenerate the PDF

Install [Typst](https://typst.app/open-source/), then run:

```sh
python3 docs/report/render_pdf.py
```

`render_pdf.py` converts the limited CommonMark subset used by the report directly
to Typst and compiles the PDF. It does not create or consume HTML.
