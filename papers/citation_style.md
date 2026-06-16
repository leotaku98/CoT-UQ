# Citation Style Rules

Derived from the user's own `.bib` files. Follow these exactly when adding new references.

## Key naming convention

`{firstauthorlastname}{year}{1-2 keyword from title}` — all lowercase, no underscores.

Examples:
- `lakshminarayanan2017simplescalablepredictiveuncertainty`
- `cohen2021gradient`
- `manakul2023selfcheckgpt`
- `wei2022chain`

## Entry types

| Venue type | Entry type |
|---|---|
| Conference (ICLR, NeurIPS, ICML, EMNLP, ACL, CVPR, etc.) | `@inproceedings` |
| Journal (Nature, TMLR, Neural Computation, etc.) | `@article` |
| Book | `@book` |
| arXiv preprint | `@preprint` with `archivePrefix = {arXiv}` and `Eprint = {xxxx.xxxxx}` |

## Required fields for @inproceedings

```bibtex
@inproceedings{key,
  author    = {First Last and First Last and First Last},
  title     = {Title With {ACRONYMS} Preserved},
  booktitle = {Full Venue Name (not abbreviated)},
  year      = {YYYY},
  pages     = {NNN--NNN},      % include if known
  address   = {City, Country}, % always include; use "Online" for virtual
  publisher = {Publisher},     % include if known
}
```

## Required fields for @article

```bibtex
@article{key,
  author    = {First Last and First Last},
  title     = {Title},
  journal   = {Full Journal Name},
  year      = {YYYY},
  volume    = {N},
  number    = {N},
  pages     = {NNN--NNN},
}
```

## Required fields for @preprint

```bibtex
@preprint{key,
  title         = {Title},
  author        = {First Last and First Last},
  archivePrefix = {arXiv},
  Eprint        = {XXXX.XXXXX},
  year          = {YYYY},
}
```

## Title formatting rules

- Use `{...}` around acronyms and words that must stay capitalised:
  `{DICE}`, `{SGD}`, `{B}ayesian`, `{FiLM}`, `{LLMs}`, `{CoT}`, `{BD}`
- Use `{...}` around the first letter only to force capitalisation of a single letter:
  `{B}ayesian` → renders as "Bayesian"
- Title case throughout

## Author format

- `First Last` order (not `Last, First`)
- Multiple authors joined by ` and `
- Special characters escaped: `G\"{u}nd\"{u}z`, `D\textquotesingle Angelo`

## What NOT to include

- No `url` field
- No `doi` field
- No `abstract` field
- No `keywords` field

## Booktitle — full names (examples)

| Short | Full |
|---|---|
| NeurIPS | Advances in Neural Information Processing Systems |
| ICLR | International Conference on Learning Representations |
| ICML | Proceedings of the N-th International Conference on Machine Learning |
| EMNLP | Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP) |
| ACL | Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL) |
| NAACL | Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL) |
| CVPR | Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) |
| COLING | Proceedings of the International Conference on Computational Linguistics (COLING) |
