filename_main = main
tex_main = $(filename_main).tex
dvi_main = $(filename_main).dvi
pdf_main = $(filename_main).pdf
bbl_main = $(filename_main).bbl
blg_main = $(filename_main).blg

filename_supp = supp
tex_supp = $(filename_supp).tex
dvi_supp = $(filename_supp).dvi
pdf_supp = $(filename_supp).pdf
bbl_supp = $(filename_supp).bbl
blg_supp = $(filename_supp).blg

filename_note = note
tex_note = $(filename_note).tex
dvi_note = $(filename_note).dvi
pdf_note = $(filename_note).pdf
bbl_note = $(filename_note).bbl
blg_note = $(filename_note).blg

filename_reply = reply
tex_reply = $(filename_reply).tex
dvi_reply = $(filename_reply).dvi
pdf_reply = $(filename_reply).pdf
bbl_reply = $(filename_reply).bbl
blg_reply = $(filename_reply).blg

all:
	make main_pdflatex
#	make main_pdflatex_pdflatex_shell_escape
#	make supp_pdflatex
#	make supp_pdflatex_pdflatex_shell_escape
#	make note_pdflatex
#	make note_pdflatex_pdflatex_shell_escape
#	make reply_pdflatex
#	make reply_pdflatex_pdflatex_shell_escape

clean:
	rm -f *.toc *.out *Notes*.bib *.blg *.fls *.fdb_latexmk .DS_Store *.auxlock *.synctex* texput.log *.dvi *.*~ *.bak* *.gz .* */.* */*/.* */*/*/.* */*/*/*/.* *.tmp indent.log

clean_aux:
	rm -f *.aux

clean_pdf:
	rm -f *.pdf

clean_bbl:
	rm -f *.bbl

clean_log:
	rm -f *.log

main_pdflatex:
	pdflatex $(tex_main)
	pbibtex $(filename_main)
	pdflatex $(tex_main)
	pdflatex $(tex_main)
	make clean

main_pdflatex_shell_escape:
	pdflatex -shell-escape $(tex_main)
	pbibtex $(filename_main)
	pdflatex -shell-escape $(tex_main)
	pdflatex -shell-escape $(tex_main)
	make clean

supp_pdflatex:
	pdflatex $(tex_supp)
	pbibtex $(filename_supp)
	pdflatex $(tex_supp)
	pdflatex $(tex_supp)
	make clean

supp_pdflatex_shell_escape:
	pdflatex -shell-escape $(tex_supp)
	pbibtex $(filename_supp)
	pdflatex -shell-escape $(tex_supp)
	pdflatex -shell-escape $(tex_supp)
	make clean

note_pdflatex:
	pdflatex $(tex_note)
	pbibtex $(filename_note)
	pdflatex $(tex_note)
	pdflatex $(tex_note)
	make clean

note_pdflatex_shell_escape:
	pdflatex -shell-escape $(tex_note)
	pbibtex $(filename_note)
	pdflatex -shell-escape $(tex_note)
	pdflatex -shell-escape $(tex_note)
	make clean

reply_pdflatex:
	pdflatex $(tex_reply)
	pbibtex $(filename_reply)
	pdflatex $(tex_reply)
	pdflatex $(tex_reply)
	make clean
 
reply_pdflatex_shell_escape:
	pdflatex -shell-escape $(tex_reply)
	pbibtex $(filename_reply)
	pdflatex -shell-escape $(tex_reply)
	pdflatex -shell-escape $(tex_reply)
	make clean

