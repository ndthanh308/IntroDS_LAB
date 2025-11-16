%% CCJNL.cls 2020/08/22 version V1.0

\ProvidesClass{ccjnl}[2020/08/22 V1.0 by Melody]
\NeedsTeXFormat{LaTeX2e}
\LoadClass[twoside,twocolumn]{article}

\RequirePackage{geometry}
\RequirePackage{graphicx}
\RequirePackage{mfirstuc}
\RequirePackage{epstopdf}
\RequirePackage{booktabs}
\RequirePackage{setspace}
\RequirePackage{algorithm,algorithmic}
\RequirePackage{titlesec}
\RequirePackage{xcolor}
\RequirePackage{tikz}
\RequirePackage{anyfontsize}
\RequirePackage{wrapfig}
\RequirePackage{subfig}
\RequirePackage{enumitem}
\RequirePackage{endnotes}
\RequirePackage{listings}
\RequirePackage{amsmath,amssymb,amsthm,siunitx,makecell,subfloat,cuted}
% \RequirePackage{multicol}
\RequirePackage{caption}
\RequirePackage[multiple,bottom]{footmisc}
% \RequirePackage{dblfnote}
% \RequirePackage{auto-pst-pdf,psvectorian}
\RequirePackage[colorlinks,allcolors=blue,urlcolor=black]{hyperref}
\RequirePackage[nameinlink]{cleveref}
\RequirePackage[numbers,compress]{natbib}

\geometry{a4paper,left=1.8cm,right=1.8cm,marginparsep=2pt,top=2.5cm,bottom=2.5cm,includeheadfoot,headsep=10pt,footskip=0.4cm}
\columnsep0.7cm


% endnotes
\let\footnote=\endnote
\setlength{\abovedisplayskip}{14pt plus 1pt minus 1pt}
\setlength{\belowdisplayskip}{14pt plus 1pt minus 1pt}

% contant
\def\journal{China Communications}

\newcommand{\receiveddate}[1]{\def\CCJNLreceiveddate{#1}}
\newcommand{\reviseddate}[1]{\def\CCJNLreviseddate{#1}}
\newcommand{\Editor}[1]{\def\CCJNLEditor{#1}}

% font type
\renewcommand{\rmdefault}{ptm}
\renewcommand{\sfdefault}{phv}
\renewcommand{\ttdefault}{ppl}

% font size
\def\yihao{\fontsize{25pt}{30pt}\selectfont}\let\Huge\yihao
% \def\xiaoyi{\fontsize{24pt}{32pt}\selectfont}
\def\erhao{\fontsize{20pt}{26pt}\selectfont}\let\huge\erhao
\def\xiaoer{\fontsize{17pt}{26pt}\selectfont}\let\LARGE\xiaoer
% \def\sanhao{\fontsize{16pt}{21pt}\selectfont}
\def\xiaosan{\fontsize{14pt}{18pt}\selectfont}\let\Large\xiaosan
% \def\sihao{\fontsize{14pt}{18pt}\selectfont}
\def\xiaosi{\fontsize{12pt}{15pt}\selectfont}\let\large\xiaosi
\def\wuhao{\fontsize{11pt}{14pt}\selectfont}\let\normalsize\wuhao
\def\xiaowu{\fontsize{10pt}{12pt}\selectfont}\let\small\xiaowu
\def\liuhao{\fontsize{9pt}{11pt}\selectfont}\let\footnotesize\liuhao
\def\xiaoliu{\fontsize{8pt}{10pt}\selectfont}\let\scriptsize\xiaoliu
\def\qihao{\fontsize{6pt}{8pt}\selectfont}\let\qihao\tiny
\def\endfont{\fontsize{9pt}{11.6pt}\selectfont}

% page layout
\newcommand{\headruleheight}{0.283pt}
\newlength\headrulewidth
\setlength\headrulewidth{\textwidth}
\def\headrule{
  \rule{\headrulewidth}{\headruleheight}
}
\def\footrule{
    \rule[0pt]{\headrulewidth}{\headruleheight}
    \rule[10pt]{\headrulewidth}{\headruleheight}
}

\def\@oddhead{\leavevmode\hbox to \textwidth{\hskip-0.1pt
  \headrule\hfill \kern\z@}}\let\@evenhead\@oddhead

\def\@oddfoot{
    {\hskip-1.6pt\parbox[t]{\headrulewidth}{\footrule\vskip-10pt
    \hbox to \headrulewidth{{\footnotesize\journal}  \hfill  {\sf\small\thepage}}}
}}
\def\@evenfoot{
  \hskip-1.6pt\parbox[t]{\headrulewidth}{\footrule\vskip-10pt
    \hbox to \headrulewidth{{\small\textsf{\thepage}} \hfill {\footnotesize\journal}}}
}

% maketitle
\newlength\boxwidth
\setlength\boxwidth{\textwidth}
\addtolength\boxwidth{-6pt}
\def\titlerule{
    {{\rule{6cm}{1pt}}}
}

\def\inst#1{\textsuperscript{\fontsize{6.1pt}{8pt}\selectfont #1}}
\newcommand\corinfo[1]{\gdef\CCJNLcorinfo{\textsuperscript{\fontsize{5.2pt}{7pt}\selectfont *~}\fontsize{9pt}{12.8pt}\selectfont The corresponding author, email: #1}}


\def\CCJNLaddress{}
\def\CCJNLcorinfo{}
\def\addsep{\par\vskip6pt}
\def\address{\@ifnextchar[{\@address}{}}
\long\def\@address[#1]#2{\g@addto@macro\CCJNLaddress{%
  \textsuperscript{\fontsize{5.2pt}{7pt}\selectfont #1~}{\fontsize{9pt}{12.8pt}\selectfont #2\\}}\par}

\def\maketitle{
  \twocolumn[
  \begin{@twocolumnfalse}
    {\begin{tikzpicture}
      \fill[fill=white] (0,0) rectangle(\textwidth,0.3);
    \end{tikzpicture}\vskip-11pt
    {\parbox[t]{\boxwidth}{\vfill\hskip-3pt\parbox{\boxwidth}{\raggedright\large\sf\scshape\textbf{\journal}}\vfill}}\vskip1cm
    \raggedright{\LARGE\sf\textbf{\@title}\par}\vskip12pt
    \titlerule\\
    {\textbf{\fontsize{10.5pt}{12pt}\selectfont\@author}}\vskip12pt
    \scriptsize\CCJNLaddress\CCJNLcorinfo
    \vskip1cm}
    % \begin{multicols}{2}
  \end{@twocolumnfalse}
  ]
  {\let\thefootnote\relax\footnotetext{\parbox[t]{0.45\textwidth}{
      {{\textbf{Received:~}\CCJNLreceiveddate}}\\
      {{\textbf{Revised:~}\CCJNLreviseddate}}\\
      {{\textbf{Editor:~}\CCJNLEditor}}\\
    }}}
}

% \AtEndDocument{\end{multicols}}

%keywords
\newcommand{\keywords}[1]{\def\CCJNLkeywords{#1}}
% abstract
\renewenvironment{abstract}{\noindent\textbf{Abstract:~}}{\\\textbf{Keywords:~}\CCJNLkeywords\par}

% section title
\renewcommand{\thesection}{\Roman{section}}
\renewcommand{\thesubsection}{\arabic{section}.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}

\gMFUnocap{in}\gMFUnocap{to}\gMFUnocap{the}\gMFUnocap{for}\gMFUnocap{on}\gMFUnocap{into}\gMFUnocap{onto}\gMFUnocap{at}\gMFUnocap{by}\gMFUnocap{within}\gMFUnocap{about}
\gMFUnocap{and}\gMFUnocap{or}\gMFUnocap{than}\gMFUnocap{until}\gMFUnocap{of}
\titleformat{\section}{\large\bf}{\thesection.}{0.5em}{\MakeUppercase}
\titleformat{\subsection}{\large\bf}{\thesubsection}{0.5em}{\capitalisefmtwords}
\titleformat{\subsubsection}{}{\thesubsubsection}{0.5em}{}
\titleformat{\paragraph}[runin]{\bf\filright}{}{0.5em}{}


\captionsetup[figure]{textfont={it,small},labelfont={bf},labelsep=period,name=Figure}
\captionsetup[table]{textfont={it,footnotesize},labelfont={footnotesize,bf},labelsep=period}
\captionsetup[algorithm]{textfont={it,footnotesize},labelfont=bf,labelsep=period}

\renewcommand{\bibsep}{0pt}

\renewcommand\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \gdef\thesection{\Alph{section}}
  \protect\vskip12pt\section*{\hfill APPENDIX \hfill}\vskip12pt
}

% footnote
\setlength{\footnotemargin}{0.5em}
\setlength{\footnotesep}{0pt}
\def\footnoterule{
  \noindent{\rule{4cm}{0.283pt}}\vskip5pt
}

% Biography
\newdimen\CCJNLdimenA
\newenvironment{CCJNLbiography}[2]{
\begin{figure}[!h]
  \vskip-26pt
    \def\CCJNLtempbiographybox{\mbox{\begin{minipage}[b][1in][c]{1.25in}\centering
    \includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{#1}
    \end{minipage}}}
    \let\CCJNLBiopar=\par
    \edef\par{\hfil\break\indent}
    \par
    \setlength\hangindent{1.2in}
    \setlength\hangafter{-10}
    \settoheight{\CCJNLdimenA}{\mbox{T}}
    \makebox[0pt][l]{\hspace{-1.5in}\raisebox{0.6\CCJNLdimenA}[0pt][0pt]{
    \raisebox{-1.25in}[0pt][0pt]{\CCJNLtempbiographybox}}}\\
    \endfont\noindent\textbf{\itshape #2} \ }
    {\relax\let\par=\CCJNLBiopar\par%
    \vskip12pt
\end{figure}
  }


\renewcommand\enoteformat{\rightskip=0pt \leftskip=0pt \parindent=0em
\leavevmode\makeenmark\raggedright\normalsize}

\renewcommand{\bibsection}{{\subsection*{References}}}
\newcommand{\biographies}{{\subsection*{Biographies}}}

\lstset{frame=single,frameround=tttt,breaklines=true,basicstyle=\scriptsize}

\setlist{itemsep=0pt, topsep=0pt}

\newtheorem{theorem}{Theorem}
\newtheorem{definition}{Definition}
\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\newtheorem{example}{Example}
\newtheorem{exercise}{Exercise}
\newtheorem{remark}{Remark}
\newtheorem{case}{Case}

%colorbox
% \RequirePackage{tcolorbox}
% \tcbuselibrary{theorems,skins,breakable}
\renewcommand{\qedsymbol}{}
% \tcolorboxenvironment{proof}{enhanced jigsaw,colframe=cyan,interior hidden,breakable,before skip=10pt,after skip=10pt}

\renewcommand{\algorithmicrequire}{ \textbf{Input:}} %Use Input in the format of Algorithm
\renewcommand{\algorithmicensure}{ \textbf{Output:}} %UseOutput in the format of Algorithm
