%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% AeroBest LaTeX class for the manuscript submitted to the         %%
%% AeroBest 2023 ECCOMAS Thematic Conference                        %%
%%                                                                  %%
%% The regular article class commands still exist and may be used.  %%
%%                                                                  %%
%% The following options have been disabled:                        %%
%% - a5paper,b5paper,letterpaper,legalpaper,executivepaper,landscape%%
%% - 10pt, 11pt                                                     %%
%% - twoside                                                        %%
%% - twocolumn                                                      %%
%% - leqno, fleqn                                                   %%
%%                                                                  %%
%% \maketitle make the first page automatically                     %%
%%                                                                  %%
%% Last Modified: Nov 4, 2022                                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{AeroBest_paper}

%%% Load 'article' class disabling some options

\DeclareOption{a5paper}{\OptionNotUsed}
\DeclareOption{b5paper}{\OptionNotUsed}
\DeclareOption{letterpaper}{\OptionNotUsed}
\DeclareOption{executivepaper}{\OptionNotUsed}
\DeclareOption{landscape}{\OptionNotUsed}

\DeclareOption{10pt}{\OptionNotUsed}
\DeclareOption{11pt}{\OptionNotUsed}

\DeclareOption{twoside}{\OptionNotUsed}

\DeclareOption{twocolumn}{\OptionNotUsed}

\DeclareOption{leqno}{\OptionNotUsed}
\DeclareOption{fleqn}{\OptionNotUsed}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions\relax

\LoadClass[a4paper,12pt,oneside,onecolumn]{article}

%%% Required packages

% Indent first paragraph after section header
% https://www.ctan.org/pkg/indentfirst
\RequirePackage{indentfirst}

%%% Define Title page

\renewcommand{\maketitle}{%
   \newpage
   \thispagestyle{plain}
   \begin{minipage}[t][1cm][t]{15.5cm}
      \vspace{-10pt}
      \includegraphics[height=1cm]{AeroBest_logo.png}
      \begin{flushright}
         \scriptsize
         \vspace{-40pt}
         AeroBest 2023 \\
         %International Conference on Multidisciplinary Design Optimization of Aircraft\\
         Lisboa, 19-21 July 2023\\
         \copyright ECCOMAS, Portugal
      \end{flushright}
      \vspace{.5cm}
   \end{minipage}
   \begin{center}%
      {% title
         \fontsize{14}{16}\selectfont
         \bfseries
         \MakeUppercase{\@title}
         \par
      }%
      \vspace{13pt}%
      {% author
         \fontsize{12}{14}\selectfont
         \bfseries\@author
         \par
      }%
      \vspace{10pt}%
      {% address
         \fontsize{11}{12}\selectfont
         \def\and{\\\vspace{13pt}}
         \normalfont\@address
         \par
      }%
   \end{center}
   \vspace{11pt}% abstract
   \noindent\textbf{Abstract.\space}{\textit\@abstract}
   \par
   \vspace{12pt}% keywords
   \noindent\textbf{Keywords:\space}\@keywords
   \par
   \vspace{18pt}%
   \newpage
}% End of \@maketitle

\def\author{\@ifnextchar[{\@getshortauthor}{\@getauthor}}
\def\@getauthor#1{\gdef\@author{#1}\gdef\@shortauthor{#1}}
\def\@getshortauthor[#1]#2{\gdef\@shortauthor{#1}\gdef\@author{#2}}

\newcommand{\address}[1]{\gdef\@address{#1}}

\renewcommand{\abstract}[1]{\gdef\@abstract{#1}}

\newcommand{\keywords}[1]{\gdef\@keywords{#1}}

%%% Define heading and corresponding page style

\newcommand{\heading}[1]{\gdef\@heading{#1}}

\def\ps@myheadings{%
   \def\@oddfoot{\normalfont\hfil\thepage\hfil}
   \let\@evenfoot\@oddfoot
   \def\@evenhead{%
      \parbox[t][12.5pt][t]{\textwidth}{%
         \centering
         \normalfont
         \fontsize{10}{10}\selectfont
         \@heading\\
         \vspace{-.2cm}
         \rule{\textwidth}{0.5pt}%
       }
    }%
    \let\@oddhead\@evenhead
    \let\@mkboth\@gobbletwo
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
}

%%% Sectioning divisions

\renewcommand\section{%
  \@startsection{section}% name
    {1}    % level
    {0pt}  % indent
    {12pt} % beforeskip, default {-3.5ex plus -1ex minus -0.2ex}
    {6pt}  % afterskip, default {2.3ex plus.2ex}
    {\normalsize\bfseries\MakeUppercase}% style, default {\normalfont\Large\bfseries}
  }

\renewcommand\subsection{%
  \@startsection{subsection}% name
    {2}    % level
    {0pt}  % indent
    {12pt} % beforeskip, default {-3.5ex plus -1ex minus -0.2ex}
    {6pt}  % afterskip, default {2.3ex plus.2ex}
    {\normalsize\bfseries}% style, default {\normalfont\large\bfseries}
  }

%%% Captions of figures and tables

\renewcommand{\@makecaption}[2]{%
   \vskip\abovecaptionskip
   \sbox\@tempboxa{\normalsize\fontsize{10}{12}\selectfont #1: #2}%
   \ifdim \wd\@tempboxa >\hsize
      \normalsize\fontsize{10}{12}\selectfont #1: #2\par
   \else
      \global \@minipagefalse
      \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
   \fi
   \vskip\belowcaptionskip}

%%% References

\renewenvironment{thebibliography}[1]
  {%\newpage
   \section*{\MakeUppercase\refname\@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}}%
   \list{\@biblabel{\@arabic\c@enumiv}}%
        {\settowidth\labelwidth{\@biblabel{#1}}%
         \leftmargin\labelwidth
         \advance\leftmargin\labelsep
         \@openbib@code
         \usecounter{enumiv}%
         \let\p@enumiv\@empty
         \renewcommand\theenumiv{\@arabic\c@enumiv}}%
   \sloppy
   \clubpenalty4000
   \@clubpenalty \clubpenalty
   \widowpenalty4000%
   \sfcode`\.\@m}
  {\def\@noitemerr
   {\@latex@warning{Empty `thebibliography' environment}}%
   \endlist}

%%% Margins

% These are exact settings for a A4 page with top margin of
% 30 mm, bottom margin of 30 mm, left and right margins of 25 mm,
% printable area 237 X 160 mm.

\setlength{\topmargin}{-5.4mm}
\setlength{\oddsidemargin}{0mm}
\setlength{\evensidemargin}{0mm}
\setlength{\marginparwidth}{0mm}
\setlength{\marginparsep}{0mm}

%%% Headings

\setlength{\headheight}{0.0mm}
\setlength{\headsep}{10.0mm}

%%% Text box

\setlength{\textwidth}{160mm}  % 210-25-25 
\setlength{\textheight}{237mm} % 297-30-30

%%% Spacings in floats

% distance between top or bottom of floats (figures and tables) and text
\setlength{\textfloatsep}{12.0pt plus 1.0pt minus 2.0pt} % 20.0pt plus 2.0pt minus 4.0pt

% distance between floats (top or bottom) inserted inside the page text (using h) and text
\setlength{\intextsep}{12.0pt plus 2.0pt minus 2.0pt} % 14.0pt plus 4.0pt minus 4.0pt

% added distance (flexible space) before/after a displayed equation
\setlength{\abovedisplayskip}{6.0pt plus 2.0pt minus 3.0pt} % 12.0pt plus 3.0pt minus 7.0pt
\setlength{\abovedisplayshortskip}{0.0pt plus 2.0pt} % 0.0pt plus 3.0pt
\setlength{\belowdisplayskip}{6.0pt plus 2.0pt minus 3.0pt} % 12.0pt plus 3.0pt minus 7.0pt
\setlength{\belowdisplayshortskip}{6.0pt plus 2.0pt minus 2.0pt} % 6.5pt plus 3.5pt minus 3.0pt

% spacing above and below figure/table caption
\setlength{\abovecaptionskip}{6.0pt} % normal font size (12 pt)
\setlength{\belowcaptionskip}{0.0pt} % 0.0 pt

%\newcommand{\printvalues}{abovedisplayskip=\the\abovedisplayskip; belowdisplayskip=\the\belowdisplayskip; abovedisplayshortskip=\the\abovedisplayshortskip; belowdisplayshortskip=\the\belowdisplayshortskip}

%%% Paragraph indent

\setlength{\parindent}{5mm} % 17.62482pt

%%% Spacings in lists

% rubber space between first item and preceding space
\setlength{\topsep}{6.0pt plus 2.0pt minus 3.0pt}
% ruber space between successive items
%\setlength{\itemsep}{5.0pt plus 2.5pt minus 1.0pt}

%\newcommand{\printvalues}{topsep=\the\topsep; itemsep=\the\itemsep; parsep=\the\parsep; partopsep=\the\partopsep}

%%% Remove page numbering

\pagenumbering{gobble}

%%% Set default page and maketitle

\AtBeginDocument{%
  \pagestyle{myheadings}
  \maketitle
  \let\maketitle\relax
}

\endinput

