\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{alt2024}[2024/01/01 v1.0 ALT 2024 Proceedings]
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

\jmlrvolume{}
\jmlryear{}
\jmlrproceedings{PMLR}{}
\jmlrpages{}
\renewcommand*{\@titlefoot}{
}

\ifanonsubmission
 \newcommand{\altauthor}[1]{\author{#1}}
 %\author{author names withheld}
 %\editor{ te }
 \jmlrworkshop{ }
 %\renewcommand{\acks}[1]{}
\else
 \newcommand{\altauthor}[1]{\author{#1}}
 \iffinalsubmission
  %\editors{Claire Vernade and Daniel Hsu}
  %\jmlrworkshop{35th International Conference on Algorithmic Learning Theory}
 \fi
\fi



