\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{hjms}[15/10/2003 Hacettepe Journal of Mathematics and Statistics Class 
()]

\newif\if@pdf
\@pdffalse
\DeclareOption{pdf}{\@pdftrue}

\newif\if@qnumis
\@qnumisfalse
\DeclareOption{eqnumis}{\@qnumistrue}

\newif\if@qnumps
\@qnumpsfalse
\DeclareOption{eqnumps}{\@qnumpstrue}

\newif\if@galley
\@galleyfalse
\DeclareOption{galley}{\@galleytrue}

\newif\if@memorial
\@memorialfalse
\DeclareOption{memorial}{\@memorialtrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{amsart}}

\ProcessOptions

\LoadClass[10pt]{amsart}

\usepackage{latexsym}
\usepackage{amsfonts}
\usepackage{epsfig}
\usepackage[cp850]{inputenc}
\usepackage[english]{babel}
\usepackage{ifthen}
\usepackage{graphicx}
\usepackage{float}
\usepackage{epstopdf}

\usepackage{amstext}
\usepackage{syntonly}

\usepackage{amsthm}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{eucal}

\swapnumbers
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{claim}[theorem]{Claim}
\newtheorem{axiom}[theorem]{Axiom}
\newtheorem{conjecture}[theorem]{Conjecture}
\newtheorem{fact}[theorem]{Fact}
\newtheorem{statement}[theorem]{Statement}
\newtheorem{hypothesis}[theorem]{Hypothesis}
\newtheorem{assumption}[theorem]{Assumption}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{criterion}[theorem]{Criterion}
\newtheorem{principle}[theorem]{Principle}
\newtheorem{property}[theorem]{Property}
\newtheorem{algorithm}[theorem]{Algorithm}
\newtheorem{result}[theorem]{Result}
\newtheorem{results}[theorem]{Results}

\theoremstyle{definition}
\newtheorem{appicationl}[theorem]{Application}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}
\newtheorem{counterexample}[theorem]{Counterexample}
\newtheorem{counterexamples}[theorem]{Counterexamples}
\newtheorem{examples}[theorem]{Examples}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{remarks}[theorem]{Remarks}
\newtheorem{note}[theorem]{Note}
\newtheorem{notes}[theorem]{Notes}
\newtheorem{case}[theorem]{Case}
\newtheorem{notation}[theorem]{Notation}
\newtheorem*{summary}{Summary}
\newtheorem{problem}[theorem]{Problem}
\newtheorem{openproblem}[theorem]{Open Problem}
\newtheorem{question}[theorem]{Question}

\newcommand{\arttitle}{}
\newcommand{\arttitlecab}{}

\setcounter{tocdepth}{0}
\newcommand{\l@arti}{\vskip0.5cm\@dottedtocline{0}{1.5em}{2.3em}}

\def\pageno=\thepage
\def\advancepageno{ifnum\pageno<0 \global\advance\pageno by -1
           \else\global\advance\pageno by 1 \fi}

\renewcommand{\@secnumfont}{\bfseries}

\renewcommand{\section}{\@startsection{section}{1}%
  \z@{.7\linespacing\@plus\linespacing}{.5\linespacing}%
  {\normalfont\large\bfseries}}

\renewcommand{\title}[2]{\renewcommand{\thefootnote}{\fnsymbol{footnote}}
\setcounter{section}{0}\setcounter{footnote}{0}
\begin{center} {\LARGE\bf #1}\end{center}\bigskip
\bigskip\thispagestyle{empty}
\renewcommand{\arttitle}{#1}\renewcommand{\arttitlecab}{#2}}

\renewcommand{\author}[2]{\begin{center} {#1}
\end{center}\vspace{5.5ex}\addcontentsline{toc}{arti}{\arttitle. By #2}\markboth{ #2}
{\arttitlecab}
%\renewcommand{\thefootnote}{\arabic{\footnote}}
}

\newcommand{\received}[1]{\noindent\small Received #1}

\newcommand{\revised}[1]{\small\ : Revised #1}

\newcommand{\accepted}[1]{\small\ : Accepted #1 \vspace{5.5ex} }

\newcommand{\comby}[1]{\small\begin{center} Communicated by #1 \end{center}}

\renewcommand{\maketitle}{\relax}

%\def\@bibliosize{\small}
%\def\@historysize{\small}
%\def\@keywordsize{\small}
%\def\@overaddressskip{2pt}
%\def\@titlesize{\Large\bfseries}
%\def\@authorsize{\large}
%\def\@keywordheading{{\it Key words: \ }}
%\def\@addressstyle{\small\itshape}
%\def\@captionsize{\small}
%\def\@tablecaptionsize{\@captionsize}
%\def\@figurecaptionsize{\@captionsize}
%\def\@tablesize{\small}
%\def\@keywordwidth{.8\textwidth}
\def\@abstractwidth{.8\textwidth}

\renewenvironment{abstract}{
\begin{center}\bf Abstract\end{center}\smallskip
\begin{center}\begin{minipage}[c]{\@abstractwidth}\small}
{\end{minipage}\end{center}
\bigskip}

\newenvironment{keyword}{\noindent{\bf Keywords: }}%
{\endtrivlist\medskip}

\newenvironment{AMS}{\noindent{\it 2000 AMS Classification: }}%
{\endtrivlist\bigskip}


\long\def\@makealgocaption#1#2{\vskip 2ex \small
\hbox to \hsize{\parbox[t]{\hsize}{{\bf #1.} #2}}}
\newcounter{algorithm}
\def\thealgorithm{\@arabic\c@algorithm}
\def\fps@algorithm{tbp}
\def\ftype@algorithm{4}
\def\ext@algorithm{lof}
\def\fnum@algorithm{Algorithm \thealgorithm}
\def\algorithm{\let\@makecaption\@makealgocaption\@float{algorithm}}
\let\endalgorithm\end@float

%This makes the "Hacettepe Journal of Mathematics and Statistics" heading for the
%firstpage.
\newcommand{\hjmsvol}{}
\newcommand{\journalvol}[1]{\renewcommand{\hjmsvol}{#1}}
\newcommand{\hjmsno}{}
\newcommand{\journalno}[1]{\renewcommand{\hjmsno}{#1}}
\newcommand{\hjmsyear}{}
\newcommand{\journalyear}[1]{\renewcommand{\hjmsyear}{#1}}
\newcommand{\hjmsfin}{}
\newcommand{\pagefin}[1]{\renewcommand{\hjmsfin}{#1}}
\newcommand{\memname}{}
\newcommand{\memorialname}[1]{\renewcommand{\memname}{#1}}

\DeclareGraphicsRule{.wmf}{bmp}{}{}% declare GIF filename extension

\newcommand{\hjmstitle}{
\noindent
\newcommand{\ext}{wmf}
\if@pdf \renewcommand{\ext}{eps}  \fi
\begin{minipage}{6.85mm}\vspace*{-0.6ex}\hspace*{0.3em}
%\includegraphics[width=3.42mm,height=7.2mm]{h.\ext}
\end{minipage}\hspace*{-0.4em}
\begin{minipage}{75mm}
  \small\noindent Hacettepe Journal of Mathematics and Statistics
  \par\noindent\hspace*{0.07em} Volume \hjmsvol\ (\hjmsyear), \thepage\,--\,\hjmsfin
\end{minipage}
\if@galley\fbox{\Large\bf GALLEY PROOFS}\fi
\if@memorial\fbox{
\begin{minipage}{30mm}
\begin{center}
\memname\\ 
Memorial Issue
\end{center}
\end{minipage}
}\fi

\vspace{10ex}}

\if@qnumps\renewcommand{\theequation}{\thesection.\arabic{equation}}\fi

\if@qnumis\numberwithin{equation}{section}\fi

\def\ps@firstpage{\ps@plain
  \def\@oddfoot{\normalfont\scriptsize \hfil\thepage\hfil
     \global\topskip\normaltopskip}%
  \let\@evenfoot\@oddfoot
  \def\@oddhead{\@serieslogo\hss}%
  \let\@evenhead\@oddhead % in case an article starts on a left-hand page
}

\textwidth=12.5cm
\textheight=19.75cm

\endinput
