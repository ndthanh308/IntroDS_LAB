# ---------------------------------------------------------
# type "make" command in Linux to create AeroBest_paper.pdf
# ---------------------------------------------------------

# Main filename
MAIN=AeroBest_paper

# ---------------------------------------------------------

all:
	pdflatex ${MAIN}
	bibtex   ${MAIN}
	pdflatex ${MAIN}
	pdflatex ${MAIN}

clean:
	(rm -rf *.aux *.bbl *.blg *.log *.out)

veryclean: clean
	(rm -rf *.pdf)
