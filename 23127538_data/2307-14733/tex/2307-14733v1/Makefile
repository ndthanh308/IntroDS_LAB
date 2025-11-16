LATEX=pdflatex
BIBTEX=bibtex

MAIN=main.tex
RESPONSE_MAJOR=response-major.tex
RESPONSE_MINOR=response-minor.tex
HOOK=acmart-preload-hook.tex
CONFIGTEX=$(wildcard config/*.tex)
SECTIONSTEX=$(wildcard sections/*.tex)
FIGURES=$(wildcard figures/*.pdf)
GNUPLOT=$(wildcard figures/*.pdf figures/*.tex figures/*.gnuplot figures/success-rate/*.dat)
TABLES=$(wildcard tables/*.tex)
CODESNIPPETS=$(wildcard code-snippets/*)
BIBLIOGRAPHY=$(wildcard bibliography/*.bib)

LATEX_PDF=$(HOOK) $(CONFIGTEX) $(SECTIONSTEX) $(FIGURES) $(GNUPLOT) $(CODESNIPPETS) $(TABLES) $(BIBLIOGRAPHY)

SRC_OUT=src-dist.zip

all: pdf src-dist response-major response-minor


pdf: $(MAIN) $(LATEX_PDF)
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN:.tex=)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

response-major: $(RESPONSE_MAJOR) $(LATEX_PDF)
	$(LATEX) $(RESPONSE_MAJOR)
	$(BIBTEX) $(MAIN:.tex=)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

response-minor: $(RESPONSE_MINOR) $(LATEX_PDF)
	$(LATEX) $(RESPONSE_MINOR)
	$(BIBTEX) $(MAIN:.tex=)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

src-dist: $(MAIN) $(LATEX_PDF)
	rm -f $(SRC_OUT)
	zip -r $(SRC_OUT) $(MAIN) $(LATEX_PDF) Makefile

.PHONY:
clean:
	- rm -f $(SRC_OUT)
	- rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *-fig*.gnuplot comment.cut
    :x

