This file collects information useful for people encountering issues while
using the `uai2024.cls` file (based on UAI 2023 instructions).

* _Please use an up-to-date TeX installation_, such as TeXlive 2021-2023.
  TeXlive 2018 and earlier are known to contain incompatible LaTeX packages.
  Specifically, the `\section` command causes an error.

* For your submission pdf, the big patch of white space between the title
  and abstract is normal. That is where the author block will appear for
  accepted papers when option `accepted` is switched on in the `\documentclass` 
  (cf. `uai2024-template-submission.pdf` and `uai2024-template-accepted.pdf`).

* An earlier version of the class could trigger a warning when adding a
  reference inside a sectioning command (e.g., `\section{… \ref{…}`). This has
  been fixed by loading the `textcase` package in the class.

* In case you load both the `amsthm` and `tikz-external` packages, a weird
  interaction with the `hyperref` and `lastpage` packages loaded by the class
  can occur. A solution is to also load the `cleverref` package! For more
  information, see <https://tex.stackexchange.com/questions/538748>.
