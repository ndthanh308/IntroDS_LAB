ART_SRCS = article.tex \
		   intro.tex \
		   background-cra.tex \
		   riscv_jop.tex \
		   experiment.tex \
		   conclusion.tex \
		   related.tex

BIB_SRCS=bibliography.bib
EXTRA_SRCS=llncs.cls splncs04.bst
SRCS=$(ART_SRCS) $(BIB_SRCS) $(EXTRA_SRCS) Makefile

all: wasp-2023.pdf

wasp-2023.pdf: $(SRCS)
	latexmk -emulate-aux-dir -aux-directory=auxdir -output-directory=outdir --output-format=pdf $<
	cp outdir/article.pdf $@

clean:
	latexmk -emulate-aux-dir -aux-directory=auxdir -output-directory=outdir --output-format=pdf -C $<

clean-all: clean
	rm -f wasp-2023.pdf
	rm -rf auxdir
	rm -rf outdir
