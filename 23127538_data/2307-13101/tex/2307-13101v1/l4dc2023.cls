\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{l4dc2023}[2023/04/01 v2.0 L4DC 2023 Proceedings]

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

\jmlrvolume{vol 211}
\jmlryear{2023}
\jmlrproceedings{PMLR}{Proceedings of Machine Learning Research}
% \jmlryear{2022}
% \jmlrproceedings{PMLR}{NeurIPS Offline RL Workshop}

\iffinalsubmission
  \editors{N. Matni, M. Morari, G. J. Pappas}
  \jmlrworkshop{5th Annual Conference on Learning for Dynamics and Control}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Temporary workaround fixing issues with \includegraphics (concerns LaTeX2e
%% releases after October 2019) [jsieber; 2020-11-12]
\def\set@curr@file#1{\def\@curr@file{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
