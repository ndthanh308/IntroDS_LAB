PAPER = paper 
TEX = $(wildcard *.tex)
FIGS = $(wildcard figures/*.pdf figures/*.png graphs/*.pdf graphs/*.png)

.PHONY: all clean

$(PAPER).pdf: $(TEX) $(FIGS) 
	echo $(FIGS)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf
