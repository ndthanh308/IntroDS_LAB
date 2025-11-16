LATEX= pdflatex --shell-escape
BIBTEX= bibtex
XDVI = xdvi
DVIPDF= dvipdfm -p letter
PS2PDF= ps2pdf
DVIPS= dvips -t letter -P pdf
RM = /bin/rm -f
CAT = cat
ISPELL = ispell
SORT = sort

MAINDOC = main
MAINPDF = ${MAINDOC}.pdf

MAINBIB = main

PROPOSAL = proposal
PROPOSALPDF = ${PROPOSAL}.pdf

all: clean
	${LATEX} ${MAINDOC}.tex
	${BIBTEX} ${MAINDOC}.aux
	${LATEX} ${MAINDOC}.tex
	${LATEX} ${MAINDOC}.tex
	${LATEX} ${MAINDOC}.tex

proposal: clean
	${LATEX} ${PROPOSAL}.tex
	${BIBTEX} ${PROPOSAL}.aux
	${LATEX} ${PROPOSAL}.tex
	${LATEX} ${PROPOSAL}.tex

view: all
	xdg-open ${MAINDOC}.pdf

# Requires rebiber to be installed: https://github.com/yuchenlin/rebiber
rebib:
	mv ${MAINBIB}.bib ${MAINBIB}.bib.bak
	rebiber -i ${MAINBIB}.bib.bak -o ${MAINBIB}.bib

# Lint for spelling + grammar
# Requires textidote to be installed: https://github.com/sylvainhalle/textidote
lint:
	textidote --check en --output html ${MAINDOC}.tex > report.html || echo ""

spell: *.tex
	@for file in $?; do aspell --lang=en_US --mode=tex -c $$file; done

clean:
	rm -f *out *aux *bbl *blg *log *.tex~ *.dvi *.fdb_lat* *.fls *.glo *.acn *.acr *.alg *.glg *.gls *.ist
	rm -f *converted-to.pdf
	rm -f ${MAINDOC}.pdf ${PROPOSAL}.pdf
	rm -Rf auto/
	rm -Rf _minted-paper/ 
