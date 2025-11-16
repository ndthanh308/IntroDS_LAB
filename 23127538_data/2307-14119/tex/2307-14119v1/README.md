## About

Author support service provides LaTeX style files and `.tex` file templates designed for *ECAI* conference proceedings.

## Contents

The following files are given in the repository (or directly in `.zip` archive):

- `ecai.cls` - LaTeX style file designed for *ECAI* conference proceedings. Please do not change it.
  This file is already loaded in the respective template file;
- `ecai-sample-and-instructions.pdf` - instructions for the preparation of a camera-ready paper in LaTeX.
  This document contains useful information regarding the structure of your document,
  proper tagging style, layout features, etc;
- `ecai-sample-and-instructions.tex` - the main template file should be used for article preparation and
  the source file for the instructions paper `ecai-sample-and-instructions.pdf` simultaneously;
- `ecai.bst`, `ecai.bib` - BibTeX related files. If your bibliography is structured in BibTeX format,
  loading your `*.bib` file and provided BibTeX style `ecai.bst` allows you to get the final format of the bibliography.
  Please note that `bibtex` program should be used to generate the `*.bbl` file. `ecai.bib` is the minimal sample of `*.bib` file.
- `ecai-sample-and-instructions.bbl` is a sample bibliography file created by BibTeX using the `ecai.bst` file.

## Setup

- Clone the repository or download the `.zip` archive;
- Read the instructions (`ecai-sample-and-instructions.pdf`) for the preparation of your LaTeX document;
- Use the template file `ecai-sample-and-instructions.tex` to prepare your manuscript.

## Bug reports

Please submit bug report, issues or feature requests to `latex-support@vtex.lt`.
