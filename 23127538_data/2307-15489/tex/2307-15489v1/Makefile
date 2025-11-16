MAIN = ParKirSch23_preprint

.PHONY: default clean

default:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	rm -f *.blg
	rm -f *.aux
	rm -f *.log
	rm -f *.out

clean:
	latexmk -c $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml $(MAIN).thm $(MAIN).pdf

