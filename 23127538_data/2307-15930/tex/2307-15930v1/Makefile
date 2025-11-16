export SHELL := /bin/bash -o pipefail

SRC = *.tex

MAIN = main
EXTRAS = submission supplementary_material
DEFAULT = $(MAIN)

.PHONY: all
all: $(MAIN).pdf #$(foreach e,$(EXTRAS),$(e).pdf)

.PHONY: default
default: $(DEFAULT).pdf

.PHONY: FORCE
%.pdf: %.tex ./bibdatabase.bib FORCE
	latexmk $(OPT) --synctex=1 -shell-escape -pdf $*

.PHONY: continuous
continuous: default
	OPT="-pvc -halt-on-error" $(MAKE) default

submission.pdf: $(MAIN).pdf
	appl=$$(grep '\newlabel{app:artifact-link}{{A}{' $(MAIN).aux | cut -c '34-35'); \
	qpdf $< --pages . 1-$$(echo $$appl - 1 | bc) -- $@

supplementary_material.pdf: $(MAIN).pdf
	appl=$$(grep '\newlabel{app:artifact-link}{{A}{' $(MAIN).aux | cut -c '34-35'); \
	qpdf $< --pages . $$appl-z -- $@

view:
	evince $(MAIN).pdf &

.PHONY: clean
clean:
	latexmk -c

.PHONY: distclean
distclean:
	latexmk -C
