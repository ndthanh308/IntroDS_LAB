PAPER = main
TEX = $(wildcard *.tex)
BIB = main.bib
FIGS = $(wildcard figures/*.pdf figures/*.png graphs/*.pdf graphs/*.png)

.PHONY: all clean

all: $(PAPER).pdf

$(PAPER).pdf: $(TEX) $(BIB) $(FIGS) jpaper.cls
	echo $(FIGS)
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf $(ABSTRACT).pdf

