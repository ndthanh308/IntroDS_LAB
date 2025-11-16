dvi : paper_zgal_mgas.tex

	rm -f paper_zgal_mgas.ps
	rm -f paper_zgal_mgas.pdf

	latex paper_zgal_mgas.tex
	bibtex paper_zgal_mgas
	latex paper_zgal_mgas.tex
	latex paper_zgal_mgas.tex
	dvips -Ppdf -G0 -o paper_zgal_mgas.ps paper_zgal_mgas.dvi 
	ps2pdf13 -sPAPERSIZE=a4 paper_zgal_mgas.ps paper_zgal_mgas.pdf
	#dvips paper_zgal_mgas.dvi -o

	rm -f *.aux *.lof *.lot *.toc 
	rm -f paper_zgal_mgas.ps paper_zgal_mgas.dvi

# produces ONLY .pdf versions
