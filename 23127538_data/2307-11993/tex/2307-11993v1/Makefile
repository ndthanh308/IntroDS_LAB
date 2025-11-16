NAME=main
SRC=$(wildcard *.tex */*.tex)

BIB=$(wildcard *.bib */*.bib)
STY=$(wildcard *.sty *.cls)

FIGS=$(wildcard figures/*.pdf)

$(NAME).pdf: $(SRC) $(BIB) $(STY) $(FIGS) $(wildcard *.yaml)
	pdflatex $(NAME)
	bibtex $(NAME)
	pdflatex $(NAME)
	bibtex $(NAME)
	pdflatex $(NAME)

getbib: $(HOME)/tex/bib/master.bib $(HOME)/tex/bib/fsl-macros.tex $(HOME)/tex/bib/acmart.cls
	(cd $(HOME)/tex/bib && cvs update)
	-mkdir -p template
	cp -av $(HOME)/tex/bib/fsl-macros.tex template
	cp -av $(HOME)/tex/bib/master.bib template
#	cp -av $(HOME)/tex/bib/acmart.cls .

# Erez rules to facilitate easier reviewing via diffs
ezdiff:
	-mv -f diffs.txt diffs.txt.old
	-chmod u+w diffs.txt.old
	-ezgit diff `lsr-tex $(NAME)` Makefile $(BIB) > diffs.txt
	chmod 444 diffs.txt
	diffstat diffs.txt

show: $(NAME).pdf
	open $?

check:
	perl ${HOME}/proj/fsl/bib/consistify ${SRCS} ${SRC}

clean: FRC
	$(RM) $(RMFLAGS) *~ .*~ \#* core
	$(RM) $(RMFLAGS) *.dvi $(NAME).ps temp.ps *.lj *.lj4 $(NAME).bbl
	$(RM) $(RMFLAGS) $(NAME).ttt $(NAME).fff $(NAME).pdf $(NAME).txt
	$(RM) $(RMFLAGS) *.log *.aux *.toc *.lof *.blg *.lot *.cb $(NAME).out
	$(RM) $(RMFLAGS) slides2.ps slides4.ps *.bibdiff

FRC:
