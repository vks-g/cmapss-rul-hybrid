# Phase 1 report

[`main.pdf`](main.pdf) is the built report (IEEE conference format).

- `main.tex` sets up the IEEEtran template and inputs one file per section from `sections/`, in report order.
- Each section file has one owner, assigned through its GitHub issue. The report's Author Contributions section lists who wrote what.
- `refs.bib` holds every reference cited in the sections.
- Figures are read from `../../results/figures/`, which the notebooks write. Build inside a full checkout of the repository.

## Build

From this folder, with [Tectonic](https://tectonic-typesetting.github.io/) (it runs BibTeX itself):

```sh
tectonic main.tex
```

Or with a TeX Live install:

```sh
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Rebuild `main.pdf` and commit it together with any change to a section or to `refs.bib`.
