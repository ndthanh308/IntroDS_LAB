NAME = main
TRNAME = TR
LATEX = pdflatex
BIBTEX = bibtex
TMPDIR = tmp
# WARNING: Must run latex paper.tex one time before making bbl will work
all:
	$(LATEX) $(NAME).tex
	$(BIBTEX) $(NAME)
	$(LATEX) $(NAME).tex
	$(LATEX) $(NAME).tex
tr:
	$(LATEX) $(TRNAME).tex
	$(BIBTEX) $(TRNAME)
	$(LATEX) $(TRNAME).tex
	$(LATEX) $(TRNAME).tex
clean:
	rm -f *.blg *.bbl *.aux *.log *.pdf *.dvi *.vtc *.out *~
