DOC=modal_abs_vmm_paper

export TEXINPUTS=.:

VIEW=$(shell if command -v evince >/dev/null ; then echo evince ; \
             else echo open ; fi)

.PHONY: all view loop clean export import check archive

all:
	latexmk -pdf $(DOC)

view: all
	 $(VIEW) $(DOC).pdf

loop:
	latexmk -pdf -pvc $(DOC)

clean::
	git clean -fX

export: all
	scp -p -B -C $(DOC).pdf $(WWW)

import:
	cp $(HOME)/tex/bib/english.bib .

CHECK := $(shell ls *.tex \
  | grep -vw -e "\(macros\|iris\|listings\)")

check:
	@ for f in $(CHECK) ; do \
	  aspell --mode=tex --lang=en_US --encoding=utf-8 \
	         --home-dir=. --personal=.aspell.en.pws \
	         check $$f ; \
	done

ARCHIVE := cpp23cpp_paper-p49-p

SOURCES  := \
  $(wildcard *.tex) \
  $(wildcard *.sty) \
  $(wildcard *.bib) \
  $(wildcard *.cls) \
  $(wildcard *.bst) \
  $(wildcard *.ml) \
  $(wildcard *.bbl) \
  $(wildcard abstract.txt) \

archive: all
	rm -rf $(ARCHIVE) $(ARCHIVE).zip
	mkdir $(ARCHIVE)
	cp $(SOURCES) $(ARCHIVE)
	zip -r $(ARCHIVE) $(ARCHIVE)
	cd $(ARCHIVE) && \
	  pdflatex cpp_paper.tex && \
	  bibtex cpp_paper && \
	  pdflatex cpp_paper.tex && \
	  pdflatex cpp_paper.tex
	unzip -l $(ARCHIVE).zip
