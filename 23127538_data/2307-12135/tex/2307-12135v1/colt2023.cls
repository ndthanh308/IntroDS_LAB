\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{colt2023}[2023/01/01 v1.0 COLT 2023 Proceedings]
\newcommand{\todo}{\color{red}}

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[pmlr]{jmlr}

\jmlrvolume{vol {195}}
\jmlryear{2023}
\jmlrproceedings{PMLR}{Proceedings of Machine Learning Research}

\ifanonsubmission
 \newcommand{\coltauthor}[1]{}
 \author{author names withheld}
 \editor{Under Review for COLT 2023}
 \jmlrworkshop{36th Annual Conference on Learning Theory}
 \renewcommand{\acks}[1]{}
\else
 \newcommand{\coltauthor}[1]{\author{#1}}
 \iffinalsubmission
  \editors{Gergely Neu and Lorenzo Rosasco}
  \jmlrworkshop{36th Annual Conference on Learning Theory}
 \fi
\fi



