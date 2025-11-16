Documento = main
Section = sections

IsBibliography = not-set

ifeq ($(IsBibliography), not-set)
	IsBibliography = yes
endif

DocumentoTex= $(Documento).tex
DocumentoLib= style.sty
DocumentoPdf= $(Documento).pdf

SectionsTex= $(Section).tex
SectionsClean= $(Section).*~

Images = $(wildcard Figures/*.pdf)

PlotsTikz = $(wildcard Plots/*.tikz Plots/*/*.tikz)
PlotsData= $(wildcard Plots/*/*.dat Plots/*/*/*.dat)
PlotsPdf= $(patsubst %.tikz, %.pdf, $(PlotsTikz))
PlotsClean= $(wildcard Plots/*/*.log Plots/*/*.out Plots/*/*.aux Plots/*/*.*~) $(wildcard Plots/*.aux Plots/*.log Plots/*.out Plots/*.*~)

TablesTikz = $(wildcard Tables/*.tikz Tables/*/*.tikz)
TablesData= $(wildcard Tables/*/*.dat Tables/*/*/*.dat)
TablesPdf= $(patsubst %.tikz, %.pdf, $(TablesTikz))
TablesClean= $(wildcard Tables/*/*.log Tables/*/*.aux Tables/*/*.*~) $(wildcard Tables/*.aux Tables/*.log Tables/*.*~)

FileClean = *.aux *.log *.nav *.out *.toc *snm *~ *.xml *blx.bib *.blg *.bcf *.spl *.fls *.fdb_latexmk *.nlo *.sta *.dvi *.synctex.gz
FileDistClean = $(DocumentoPdf)

.PHONY: build rebuild clean distclean

build: $(DocumentoPdf)
	@make clean

rebuild: 
	@make distclean
	@make build
	@make clean

distclean: clean
	@rm -f $(PlotsPdf)
	@rm -f $(TablesPdf)
	@rm -f $(FileDistClean)

clean:
	@rm -f $(FileClean) $(PlotsClean) $(TablesClean) $(SectionsClean)

$(DocumentoPdf): $(DocumentoTex) $(DocumentoLib) $(Images) $(PlotsPdf) $(TablesPdf) $(SectionsTex)
	@pdflatex $(DocumentoTex)
ifeq ($(IsBibliography), yes)
	@bibtex $(Documento)
endif
	@pdflatex $(DocumentoTex)
	@pdflatex $(DocumentoTex)

Plots/%.pdf: Plots/%.tikz
	@cd $(dir $^); pdflatex $(shell basename $<)

Tables/%.pdf: Tables/%.tikz
	@cd $(dir $^); pdflatex $(shell basename $<)
