TARGET := submission
BIB := refs.bib

TEX_ALL := $(shell search=$(TARGET).tex; all=; \
				while [ -n "$$search" ] ; do \
					all="$$all $$search"; \
					search=`grep -E "^[^%]*input" $$search | \
						sed -En 's/.*input[^\{]*\{(.+)\}/\1.tex/p'`; \
				done; \
				echo "$$all")

FIGURES := $(shell for t in $(TEX_ALL); do \
				cat $$t | \
				grep -E '^[^%]*\\includegraphics' | \
				sed -En 's/.*includegraphics(\[.+\])?\{([^\{]*)\}.*/\2/p' | \
				grep -E -v '\.pdf$$'; \
				done)

# Package acronym Warning: Acronym `CDN'
all: $(TARGET).pdf
	@mm=`grep "Warning: Citation" $(TARGET).log | sort | uniq | wc -l | awk '{print $$1;}'`; \
	 nn=`grep "Warning: Reference" $(TARGET).log | sort | uniq | wc -l | awk '{print $$1;}'`;  \
	 oo=`grep "Warning: Label " $(TARGET).log | sort | uniq | wc -l | awk '{print $$1;}'`;  \
	 pp=`grep "Package acronym Warning: Acronym " $(TARGET).log | sort | uniq | wc -l | awk '{print $$1;}'`;  \
	 if [ $$mm -eq 0 -a $$nn -eq 0 -a $$oo -eq 0 -a $$pp -eq 0 ]; then \
	 	echo "\033[0;34mZero missing citations and references! 👏🎉\033[0m"; \
	 else \
		 echo "\033[1;33m################################"; \
		 if [ $$mm -gt 0 ]; then \
			 echo "$$mm missing citation(s):"; \
			 grep "Warning: Citation" $(TARGET).log | awk '{print "\033[0;31m   ", $$5;}' | sed "s/[\`']//g" | sort | uniq; \
		 fi; \
		 if [ $$nn -gt 0 ]; then \
			 echo "\033[1;33m$$nn missing reference(s):"; \
			 grep "Warning: Reference" $(TARGET).log | awk '{print "\033[0;31m   ", $$4;}' | sed "s/[\`']//g" | sort | uniq; \
		 fi; \
		 if [ $$oo -gt 0 ]; then \
			 echo "\033[1;33m$$oo multiply defined label(s):"; \
			 grep "Warning: Label" $(TARGET).log | awk '{print "\033[0;31m   ", $$4;}' | sed "s/[\`']//g" | sort | uniq; \
		 fi; \
		 if [ $$pp -gt 0 ]; then \
			 echo "\033[1;33m$$pp undefined acronym(s):"; \
			 grep "Package acronym Warning: Acronym" $(TARGET).log | awk '{print "\033[0;31m   ", $$5;}' | sed "s/[\`']//g" | sort | uniq; \
		 fi; \
		 echo "\033[1;33m################################\033[0m"; \
	 fi

$(TARGET).pdf: $(TEX_ALL) $(BIB)
	pdflatex $(TARGET).tex
	bibtex $(TARGET)
	pdflatex $(TARGET).tex
	pdflatex $(TARGET).tex

view:: $(TARGET).pdf
	open $(TARGET).pdf

see:: $(TARGET).dvi
	xdvi $(TARGET)

spell::
	ispell *.tex

clean::
	rm -fv *.dvi *.aux *.log *~ *.bbl *.blg *.toc *.out *.ps *.pdf *.ent parsetab.py

fresh::
	rm -fv *.dvi *.aux *.log *~ *.bbl *.blg *.toc *.ps *.pdf

distclean:: clean
	rm $(TARGET).ps

.PHONY: clean all view see spell fresh distclean nobib
