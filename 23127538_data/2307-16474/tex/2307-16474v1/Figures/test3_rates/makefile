PlotsTikz = $(wildcard ./*.tex)
PlotsPdf= $(patsubst %.tex, %.pdf, $(PlotsTikz))
PlotsClean= $(wildcard ./*.aux ./*.log ./*.out ./*.*~ ./*.fls ./*.fdb_latexmk ./*.synctex.gz)

.PHONY: build rebuild clean distclean

build: $(PlotsPdf)
	echo $(PlotsTikz)
	@make clean

rebuild: 
	@make distclean
	@make build
	@make clean

distclean: clean
	@rm -f $(PlotsPdf)

clean:
	@rm -f $(PlotsClean)

./%.pdf: ./%.tex
	pdflatex $(shell basename $<)
