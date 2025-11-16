filename=sample-authordraft

all: pdf

pdf:
	pdflatex-dev ${filename}
	- bibtex ${filename}
	pdflatex-dev ${filename}
	pdflatex-dev ${filename}
	while ( grep -q '^LaTeX Warning: Label(s) may have changed' ${filename}.log) \
	  do pdflatex-dev ${filename}; done

clean:
	$(RM)  *.log *.aux \
	*.cfg *.glo *.idx *.toc \
	*.ilg *.ind *.out *.lof \
	*.lot *.bbl *.blg *.gls *.cut *.hd \
	*.dvi *.ps *.thm *.tgz *.zip *.rpi

