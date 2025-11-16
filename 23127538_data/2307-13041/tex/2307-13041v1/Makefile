.IGNORE:

TARGET = paper

all: klk


klk: 
	pdflatex     $(TARGET)
	bibtex       $(TARGET)
	pdflatex     $(TARGET)
	pdflatex     $(TARGET)

dvi:  links
	latex     $(TARGET)
	bibtex    $(TARGET)
	latex     $(TARGET)
	latex     $(TARGET)

pdf: dvi
	dvipdf     -sPAPERSIZE=letter $(TARGET)
#	dvipdf     -sPAPERSIZE=a4 $(TARGET)
	#dvipdf     $(TARGET)
#	dvips -tletter -o $(TARGET).ps $(TARGET).dvi
#	ps2pdf $(TARGET).ps

ps: dvi
	dvips     $(TARGET)

clean:
	/bin/rm -f *.dvi *.log *.aux *.toc *.bbl *blg *.end paper.synctex.gz
	/bin/rm -f $(TARGET).ps $(TARGET).pdf  $(TARGET).tex~ Makefile~ $(TARGET).out $(TARGET).bib~

