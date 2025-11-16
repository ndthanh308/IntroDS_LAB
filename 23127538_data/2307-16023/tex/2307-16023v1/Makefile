ifeq ($(MAIN_TEX),)
	MAIN_TEX := issta-2023
endif

latexcmd = pdflatex
bibcmd = bibtex

all: compile clean

staticchecks : compile check clean

compile:
	${latexcmd} -interaction nonstopmode -halt-on-error -file-line-error --shell-escape ${MAIN_TEX}.tex 
	${bibcmd} ${MAIN_TEX}
	${latexcmd} -interaction nonstopmode -halt-on-error -file-line-error --shell-escape ${MAIN_TEX}.tex 
	${latexcmd} -interaction nonstopmode -halt-on-error -file-line-error --shell-escape ${MAIN_TEX}.tex 
	${bibcmd} ${MAIN_TEX}
	${latexcmd} -interaction nonstopmode -halt-on-error -file-line-error --shell-escape ${MAIN_TEX}.tex 

check:
	- checkcites ${MAIN_TEX}.aux > checkcites-report.txt
	- chktex ${MAIN_TEX} > chktex-report.txt

clean: 
	rm -f ${MAIN_TEX}.aux
	rm -f ${MAIN_TEX}.bbl
	rm -f ${MAIN_TEX}.blg
	rm -f ${MAIN_TEX}.log
	rm -f ${MAIN_TEX}.run.xml
	rm -f ${MAIN_TEX}-blx.bib
	rm -f ${MAIN_TEX}.dvi
	rm -f ${MAIN_TEX}.ps
	rm -f ${MAIN_TEX}.pyg
	rm -f ${MAIN_TEX}.bcf
	rm -f ${MAIN_TEX}.out
	rm -rf ${MAIN_TEX}.auxlock
	rm -rf ${MAIN_TEX}.fls
	rm -rf ${MAIN_TEX}.fdb_latexmk
	rm -rf __latexindent_temp.*
	rm -rf .texpadtmp
	# rm -rf _minted-main
	rm -rf ${MAIN_TEX}.pdf_original
	rm -rf *.tex~
	rm -rf sec/*.tex~
	rm -rf *.bib.blg
	rm -rf missfont.log
	rm -rf tmp/*

distclean: squeaky
	rm -rf ${MAIN_TEX}.pdf
	rm -rf ${latex-diff-filename}.pdf
