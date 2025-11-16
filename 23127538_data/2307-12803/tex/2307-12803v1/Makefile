all: main clean

main:
	latexmk -pdf main.tex

format:
	bibtool --preserve.key.case=on --print.use.tab=off --print.align.key=off --print.line.length=100 --suppress.initial.newline=on -r sort_fld -r braces -s bibliography.bib > bibliography-new.bib && mv bibliography-new.bib bibliography.bib

clean:
	rm -f *.log *.lof *.lot *.toc *~ *.aux *.fls *.out *.bbl *.blg *.synctex.gz *.maf *.mtc .mtc0 *.snm *.nav *.glo *.glsdefs *.ist *.vrb *.bcf *.run.xml *.glg *.gls *.acn *.acr *.alg *.lol *.fdb_latexmk

cleanall: clean
	rm -f main.pdf
