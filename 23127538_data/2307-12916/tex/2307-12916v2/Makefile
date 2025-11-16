INAME = main
ONAME = lood
EC_INAME = ec24
EC_ONAME = lood-ec24
TEX_OPTIONS = -cnf-line "max_print_line = 10000" -halt-on-error
default:
	pdflatex $(TEX_OPTIONS) -jobname=$(ONAME) $(INAME).tex
	bibtex $(ONAME).aux
	pdflatex $(TEX_OPTIONS) -jobname=$(ONAME) $(INAME).tex
	pdflatex $(TEX_OPTIONS) -jobname=$(ONAME) $(INAME).tex
	pdflatex $(TEX_OPTIONS) -jobname=$(ONAME) $(INAME).tex
ec24:
	pdflatex $(TEX_OPTIONS) -jobname=$(EC_ONAME) $(EC_INAME).tex
	bibtex $(EC_ONAME).aux
	pdflatex $(TEX_OPTIONS) -jobname=$(EC_ONAME) $(EC_INAME).tex
	pdflatex $(TEX_OPTIONS) -jobname=$(EC_ONAME) $(EC_INAME).tex
	pdflatex $(TEX_OPTIONS) -jobname=$(EC_ONAME) $(EC_INAME).tex
clean:
	rm -f *.{aux,bbl,blg,log,out,toc,vtc,cut}
