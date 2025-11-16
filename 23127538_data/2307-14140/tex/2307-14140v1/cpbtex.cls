\NeedsTeXFormat{LaTeX2e}[2007/10/19]
\ProvidesClass{cpbtex}[2014/11/12 v1.0 LaTeX Template for Chinese Physics B]

\LoadClass[a4paper]{article}[2007/10/19]

\RequirePackage{amsmath,amsfonts,amssymb,bm,CJK,ccmap}
\RequirePackage{graphics,epstopdf,ifpdf}
\RequirePackage{mathrsfs,textcomp,multicol}
\RequirePackage{indentfirst,fancyhdr,upgreek}
\RequirePackage[columnwise]{lineno}
\RequirePackage[compress,nospace]{cite}
\RequirePackage[bookmarksnumbered=true,bookmarksopen=true,colorlinks=true,pdfborder=001,
            urlcolor=blue,linkcolor=blue,anchorcolor=blue,citecolor=blue]{hyperref}
            
\newcommand{\ucite}[1]{$^{\text{\!\cite{#1}}}$} % reference citation
%================================== page setting =================================================%
\renewcommand{\baselinestretch}{1.5}    % set the line spacing
\renewcommand{\thesection}{\arabic{section}.\hspace{-.4cm}}
\renewcommand{\thesubsection}{\arabic{section}.\arabic{subsection}.\hspace{-.4cm}}
\renewcommand{\thesubsubsection}{\arabic{section}.\arabic{subsection}.\arabic{subsubsection}.\hspace{-.3cm}}
\footskip=45pt  \headsep=4truemm \topmargin=-0.5cm \oddsidemargin=0pt \evensidemargin=0pt
\textwidth=170truemm   % set page width
\textheight=250truemm  % set page height
\parindent=19pt        % set page indent
\def\thefootnote{}  %% remove the footnote number label


\endinput
%%
%% End of file `article.cls'.
