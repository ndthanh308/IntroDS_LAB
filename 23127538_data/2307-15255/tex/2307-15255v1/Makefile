all:
	pdflatex paper

full:
	pdflatex -shell-escape paper
	bibtex paper
	pdflatex -shell-escape paper
	pdflatex -shell-escape paper

clean:
	rm -f paper.pdf *.out *.aux *.cut *.log *.bbl *.blg
