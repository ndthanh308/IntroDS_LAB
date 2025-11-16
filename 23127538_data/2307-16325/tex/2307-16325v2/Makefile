######################################################################
#
#   File          : Makefile
#   Author(s)     : Adwait Nadkarni <apnadkar@ncsu.edu>
#   Description   : Makefile for Weir Oalkand 2016
#
#   Last Modified : Tue Nov  3 13:09:17 EST 2015
#   By            : Adwait Nadkarni <apnadkar@ncsu.edu>
#
#   Copyright (c) 2011 North Carolina State University
#
######################################################################

FILE=main
all: $(FILE).pdf
.PHONY: clean
.PHONY: .FORCE
clean:
	rm *.aux *.blg *.out *.bbl *.log *.synctex *.synctex.gz $(FILE).pdf

$(FILE).pdf: $(FILE).tex .FORCE
	pdflatex $(FILE)
	pdflatex $(FILE)
	bibtex $(FILE)
	pdflatex $(FILE)
	pdflatex $(FILE)