PAPER = main
REFERENCE = main
JUNK =*.log *.synctex\(busy\) *.xml *.bcf *.toc *.snm *.nav *.fls *.fdb_latexmk *.synctex.gz *.spl *.ps *.dvi  *.out *.aux *.backup *~ *.blg
COUNTING_TOOL = ./texcount.pl
TEMP = 0
MY_VAR=$(shell echo whatever)

# Define the tools we are going to use

all: pdf

# Rules for various parts of the target
pdf: $(PAPER).tex
	pdflatex $(PAPER).tex
	bibtex $(PAPER).aux
	pdflatex $(PAPER).tex
	pdflatex $(PAPER).tex
	#rm -f $(JUNK)

clean:
	rm -f $(JUNK) $(PAPER).pdf


