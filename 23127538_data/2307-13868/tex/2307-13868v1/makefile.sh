LATEX = xelatex
TEX = discriminability
BIBTEX = bibtex

default: $(TEX).tex
	$(RM) -f *.aux *.blg *.dvi *.log *.toc *.lof *.lot *.cb *.bbl *.brf *.out $(TEX).ps;
	$(LATEX) $(TEX).tex; $(BIBTEX) $(TEX); $(LATEX) $(TEX).tex; $(LATEX) $(TEX).tex;
	open $(TEX).pdf &

clean:
	$(RM) -f *.aux *.blg *.dvi *.log *.toc *.lof *.lot *.cb *.bbl *.brf *.out $(TEX).ps *~;

check:
	@echo "Passing the check will cause make to report Error 1.";
	$(LATEX) $(TEX)  | grep -i undefined;
