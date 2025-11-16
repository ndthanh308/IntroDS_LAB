\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hld2023}[2023/02/04 v1.0 HLD 2023 Style File]
% v2022: S. Stich
% v2021: S. Stich
% v2020: S. Stich
% v2019: S. Sra

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

\LoadClass[wcp]{jmlr}

% modified so that header doesn't appear
\jmlrvolume{}
\jmlryear{}
\jmlrproceedings{wcp}{HLD 2023: 1st Workshop on High-dimensional Learning Dynamics}


\ifanonsubmission
 \newcommand{\hldauthor}[1]{}
 \author{author names withheld\\[2em] Under Review for the Workshop on High-dimensional Learning Dynamics, 2023}
 \jmlrworkshop{}
 \renewcommand{\acks}[1]{}
\else
 \newcommand{\hldauthor}[1]{\author{#1}}
 \iffinalsubmission
  \editors{}
  \jmlrworkshop{}
 \fi
\fi

% patch for includegraphics not working with newer latex versions
\makeatletter
 \let\Ginclude@graphics\@org@Ginclude@graphics 
\makeatother





