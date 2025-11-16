% Copyright Javier Sánchez-Monedero.
% Please report bugs and suggestions to (jsanchezm at uco.es)
%
% This document is released under a Creative Commons Licence 
% CC-BY-SA (http://creativecommons.org/licenses/by-sa/3.0/) 
%

\ProvidesClass{reviewresponse}[2014/01/29]
\NeedsTeXFormat{LaTeX2e}

\DeclareOption*{\InputIfFileExists{\CurrentOption}{}{%
  \PassOptionsToClass{\CurrentOption}{article}}}
\ProcessOptions \relax
\LoadClass{article}

\RequirePackage{a4wide}
\RequirePackage{palatino}
\RequirePackage{verbatim}
\RequirePackage{color}
\RequirePackage{url}
\RequirePackage{fancyhdr}
\RequirePackage{fancybox}
\RequirePackage{fancyvrb}
%\RequirePackage[authoryear]{natbib}
\RequirePackage[square,sort,comma,numbers]{natbib}
\RequirePackage{tcolorbox}
\RequirePackage[parfill]{parskip}

%%%%%%%%%%%% headings setup %%%%%%%%%%%%
\pagestyle{fancy}
% clear styles
\fancyhf{}
% odd pages -> chapter name
\fancyhead[LO]{\leftmark} 
% even pages
% \fancyhead[RE]{\myJournal -- \myShortTitle}
\fancyhead[RE]{\myJournal}
% odd/even pages
\fancyhead[RO,LE]{\thepage}

\renewcommand{\sectionmark}[1]{\markboth{\textbf{\thesection. #1}}{}}
%\renewcommand{\subsectionmark}[1]{\markright{\textbf{\thesection. #1}}} 

% line under headings
\renewcommand{\headrulewidth}{0.6pt}
% line over foot 
%\renewcommand{\footrulewidth}{0.6pt}

% increases heading weight
\setlength{\headheight}{1.5\headheight}
\fancyfoot{}

\newcommand{\newtext}[1]{\textcolor{blue}{#1}}

% New revisor comment command
\newcounter{comments}[section]

\newcommand{\rcomment}[1]
{
\stepcounter{comments}
\addcontentsline{toc}{subsection}{Reviewer \arabic{section} Comment \arabic{comments}}
\begin{tcolorbox}[colback=blue!5,colframe=white!45!black,title=Reviewer \arabic{section} Comment \arabic{comments}]
#1
\end{tcolorbox}
}

\newcommand{\aecomment}[1]
{
\stepcounter{comments}
\addcontentsline{toc}{subsection}{AE Comment \arabic{comments}}
\begin{tcolorbox}[colback=blue!5,colframe=white!45!black,title=AE Comment \arabic{comments}]
#1
\end{tcolorbox}
}

\newcommand{\sumcomment}[1]
{
% \stepcounter{comments}
\addcontentsline{toc}{subsection}{Summary Comment}
\begin{tcolorbox}[colback=blue!5,colframe=white!45!black,title=Summary Comment]
#1
\end{tcolorbox}
}

\newcommand{\sectioncomment}[2]
{
\stepcounter{comments}
\addcontentsline{toc}{subsection}{#1 Comment \arabic{comments}}
\begin{tcolorbox}[colback=blue!5,colframe=white!45!black,title=#1 Comment \arabic{comments}]
#2
\end{tcolorbox}
}

\newcommand{\titlecomment}[2]
{
\addcontentsline{toc}{subsection}{Title and Comment}
\begin{tcolorbox}[colback=blue!5,colframe=white!45!black,title=#1]
#2
\end{tcolorbox}
}

