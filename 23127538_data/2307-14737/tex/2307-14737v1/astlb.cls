\ProvidesClass{astlb}

\newif\if@manuscript \@manuscriptfalse
\newif\if@twoside

\DeclareOption{a4paper}{\def\@papersize{a4paper}}
\DeclareOption{letterpaper}{\def\@papersize{letterpaper}}
\DeclareOption{twoside}{\@twosidetrue}
\DeclareOption{oneside}{\@twosidefalse}
\DeclareOption{manuscript}{
  \@manuscripttrue
  }

\ExecuteOptions{a4paper,twoside}
%\ExecuteOptions{letterpaper,twoside}
\ProcessOptions

\if@twoside
  \def\@sidetype{twoside}
\else
  \def\@sidetype{oneside}
\fi

\if@manuscript
 \LoadClass[\@papersize,\@sidetype,12pt]{article}
 \RequirePackage[doublespacing]{setspace}
 \RequirePackage[nolists,nomarkers]{endfloat}
 \def\pazhtablesize{\scriptsize}
\else
% \LoadClass[\@papersize,\@sidetype,twocolumn,11pt]{article}
% \renewcommand{\baselinestretch}{0.85}
 \LoadClass[\@papersize,\@sidetype,twocolumn]{article}
 \def\pazhtablesize{\footnotesize}
\fi

% Fonts
%\RequirePackage{amsfonts}
%\RequirePackage{amssymb}

% Journal info
\def\journalinfo#1#2#3#4{
  \gdef\@pazhyear{#1}
  \gdef\@pazhvolume{#2}
  \gdef\@pazhno{#3}
  \gdef\@pazhpage{#4}
  \pazhlastpage
  }
\def\pazhlastpage{
  \@ifnextchar[{\setpazhlastpage}
  {\gdef\@pazhlastpage{\@pazhpage}}}
\def\setpazhlastpage[#1]{\gdef\@pazhlastpage{#1}}
\journalinfo{2100}{1}{5}{1}[1] % - fake info

% Misc macros
\def\@pazh@back@one@line{\vspace{-\baselinestretch\baselineskip}}

% Page layout
\setlength{\columnsep}{5mm}

\if@twoside
% \usepackage[\@papersize,twoside,twosideshift=-0.1in,textheight=233mm,headheight=0pt,headsep=5mm,footskip=9mm,textwidth=173mm]{geometry}
 \usepackage[\@papersize,twoside,textheight=233mm,headheight=0pt,headsep=5mm,footskip=9mm,textwidth=170mm]{geometry}
\else
 \usepackage[\@papersize,textheight=233mm,headheight=0pt,headsep=5mm,footskip=9mm,textwidth=173mm]{geometry}
\fi
\sloppy


\RequirePackage{natbib}

% Title
\newenvironment{pazhtitle}{
  \large\bfseries
  \list{}{\rightmargin 0.05\linewidth\leftmargin\rightmargin}%
  \MakeUppercase
\item\relax}{\endlist}

% email
\def\email#1{\textsuperscript{*}\gdef\@email{#1}}
\def\@@email#1{
  \let\oldthefootnote\thefootnote
  \renewcommand\thefootnote{\relax}
  \footnotetext{\hspace{-\parindent}\textsuperscript{*}e-mail: #1}
  \let\thefootnote\oldthefootnote
  \setcounter{footnote}{0}
  }
\def\submitted#1{\gdef\@submitted{#1}}
\def\author#1{%
  \gdef\@author{#1}
  }
% MAKETITLE MACRO
\def\@maketitle{%
  \newpage
  \null
  \if@manuscript\else
   % \vskip -8mm
   \@pazh@back@one@line
   % ðéóøíá ÷ áö ÓÌÅ×Á ××ÅÒÈÕ ÓÔÒÁÎÉÃÙ
   \noindent{\small\itshape  Astronomy Letters, \@pazhyear, 
    Vol.~\@pazhvolume, 
%\textnumero~\@pazhno,
    No.~\@pazhno,
    p.~\@pazhpage--\@pazhlastpage. Translated from Pis'ma v Astronomicheskij
    Zhurnal.}\par
   % ä×Å ÌÉÎÉÉ
   \@pazh@back@one@line\vspace{7mm}
   \def\linerule{\noindent\makebox[\linewidth]{\hrulefill}}
   \linerule\par\vspace{-\baselinestretch\baselineskip}\vspace{1mm}\linerule
   % õäë
   \@pazh@back@one@line\vskip 9mm
   \vskip 12pt
%   \noindent {\small\itshape ASTL.CLS 04.02.12}
  \fi
 % îÁÚ×ÁÎÉÅ
  \begin{center}% 
    \begin{minipage}{0.885\linewidth}
      % áÄÒÅÓÁ ÓÄÅÌÁÎÙ Ó ÐÏÍÏÝØÀ footnotes; ÄÅÌÁÅÍ ÎÕÖÎÙÅ ÐÅÒÅÏÐÒÅÄÅÌÅÎÉÑ
      \let\footnotesize=\small
      \def\address##1{\textsuperscript{##1}}%
      \def\addresstext##1##2{\footnotetext[##1]{\itshape ##2}}%
      \def\thefootnote{\@arabic\c@footnote}
      \def\thempfootnote{\@arabic\c@mpfootnote}
      \let\footnoterule\relax
      \long\def\@footnotetext##1{%
        \global\setbox\@mpfootins\vbox{%
          \unvbox\@mpfootins
          \reset@font\footnotesize
          \@parboxrestore
          \centerline{
            \protected@edef\@currentlabel
            {\csname p@mpfootnote\endcsname\@thefnmark}%
            \color@begingroup
            \@makefntext{%
              \rule\z@\footnotesep\ignorespaces##1\@finalstrut\strutbox}%
            \color@endgroup}}}
      \begin{center}
        % óÏÂÓÔ×ÅÎÎÏ ÎÁÚ×ÁÎÉÅ
%        {\large\bfseries \MakeUppercase\@title\par
        {
          \renewcommand{\baselinestretch}{1.}
          \large\bfseries \@title\par
          \if@manuscript\else\@pazh@back@one@line\fi}%
        \if@manuscript\else\vskip 10mm\fi
        % á×ÔÏÒÙ
        {\bfseries
%          \if@manuscript\else\makebox[0pt][r]{\textcopyright~\number\year}~~~\fi
          \@author\par
          \vskip 1em}
      \end{center}
    \end{minipage}  % ðÒÉ ÚÁËÒÙÔÉÉ ÜÔÏÇÏ minipage ÐÏÌÕÞÁÀÔÓÑ ÁÄÒÅÓÁ
    \par\medskip{\footnotesize Accepted for publication in Astronomy Letters (\@submitted)}%
  \end{center}%
  \vskip 2mm
  \par
  \begin{pazhabstract}\the\abstract@toks\end{pazhabstract}
  \vskip 1.5em
  \if@manuscript\else\setcounter{page}{\@pazhpage}\fi
  }

% Abstract: the code to take the contents of the abstract environemnt is
% taken from tugboat.dtx
\newtoks\abstract@toks  \abstract@toks{}
\let\if@abstract\iffalse
\def\abstract{\def\@abstract@{abstract}%
  \ifx\@currenvir\@abstract@
  \else
    \TBError{\string\abstract\space is illegal:%
      \MessageBreak
      use \string\begin{\@abstract@} instead}%
      {\@abstract@\space may only be used as an environment}
  \fi
  \global\let\if@abstract\iftrue
  {\ifnum0=`}\fi
  \@abstract@getbody 
}
% endabstract calls maketitle and puts email; maketitle typesets the title
% text
\def\endabstract{\maketitle\@@email{\@email}}
%%%
\long\def\@abstract@getbody#1\end{%
  \global\abstract@toks\expandafter{\the\abstract@toks \textbf{Abstract ---}#1}%
  \@abstract@findend}
\def\@abstract@findend#1{%
  \def\@tempa{#1}%
  \ifx\@tempa\@abstract@
    \expandafter\@abstract@end
  \else
    \def\@tempb{document}%
    \ifx\@tempa\@tempb
      \TBError{\string\begin{\@abstract@}
          ended by \string\end{\@tempb}}%
        {You've forgotten \string\end{\@abstract@}}
    \else
       \global\abstract@toks\expandafter{\the\abstract@toks\end{#1}}%
       \expandafter\expandafter\expandafter\@abstract@getbody
    \fi
  \fi}
\def\@abstract@end{\ifnum0=`{\fi}%
  \expandafter\end\expandafter{\@abstract@}}
%% The way abstract is typeset
\newenvironment{pazhabstract}{
  \def\keywords##1{\medskip Key words:~~\emph{##1}\medskip}
  \def\englishkeywords##1{\medskip\emph{Keywords:}~~##1\medskip}
  \def\englishtitle##1{{\raggedright\uppercase{##1,}}}
  \def\englishauthors##1{{\raggedright\emph{by ##1}\par}}
  \small
  \list{}{\rightmargin 0.0575\linewidth\leftmargin\rightmargin}%
\item\relax}{\endlist}

% Sections
\setcounter{secnumdepth}{0}


\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {5.0mm \@plus 1ex \@minus .2ex}%
                                   {1.0mm \@plus.2ex}%
                     {\normalfont\centering\MakeUppercase}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {1.0ex\@plus 1ex \@minus .2ex}%
                                     {1.0ex \@plus .2ex}%
                                     {\centering\normalfont\itshape}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {3.25ex\@plus 1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\raggedright\itshape\normalsize}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                     {3.25ex\@plus 1ex \@minus -.2ex}%
                                     {0.5ex \@plus .2ex}%
                                     {\raggedright\slshape\small}}

% Captions
\AtBeginDocument{
  \def\figurename{Fig.}%
  \def\tablename{Table}%
}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\footnotesize {\bfseries #1.} #2}%
  \ifdim \wd\@tempboxa >\hsize
  \footnotesize {\bfseries #1.} #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

% Acknowledgements

\def\acknowledgements{\bigskip}

% References
\def\refvolume{v}
\def\refpage{p}

%\def\reference#1{\addtocounter{reference}{1}\thereference. #1\@reference}
\def\reference#1{#1\@reference}

\def\@reference{\@ifnextchar[{\@engreference}{\@rusreference}}
%\def\@rusreference#1{\space/\kern-0.2em/ {#1.}}

\def\@rusreference#1{,\space {#1.}}

%\def\@engreference[#1]#2{\space(#1) /\kern-0.2em/ #2.}

\def\@engreference[#1]#2{\space(#1), #2.}


%\def\journal#1#2#3#4{{#1, #2, \refvolume.~#3, \refpage.~#4}}
%\def\journal#1#2#3#4{{#1 ~#2, #3, (#4)}}
\def\journal#1#2#3#4{{#1 {\bf #2}, #3 (#4)}}

\def\astl{Astronomy Letters}
\def\pazh{Pis'ma v Astron. Zhurn.}

\def\nature{Nature}

\def\apj{Astrophys.\ J.}
\def\apjl{Astrophys.\ J.\ (Letters)}
\def\apjs{Astrophys.\ J.\ Suppl.\ Ser.}
\def\aap{Astron.\ Astrophys.}
\def\aaps{Astron.\ Astrophys.\ Suppl.\ Ser.}
\def\aapl{Astron.\ Astrophys.\ (Letters)}
\def\aj{Astron.\ J.}
\def\mnras{Mon. Not. R. Astron. Soc.}
\def\araa{Ann.\ Rev.\ of Astron.\ and Astrophys.}
\def\newar{New Astron. Rev.}
\def\revmp{Rev. Mod. Phys.}

\def\aca{Acta Astronomica}
\def\advspres{Advances in Space Research}
\def\asp{Astron. Soc. of the Pacific}

\def\revmp{Rev. Mod. Phys.}

\def\ptp{Progress of Theoretical Physics}

\def\pasp{PASP}
%\def\apss{Ap\&SS}
\def\apss{Astrophys. and Space Sci.}
\def\commapsp{Comments on Astrophysics and Space Physics}

\def\prl{Phys. Rev. Letters}
\def\prd{Phys. Rev. D}
\def\prc{Phys. Rev. C}
\def\plb{Physics Lett. B}
\def\jhep{J. of High Energy Phys.}
\def\jcap{J. of Cosmology and Astroparticle Phys.}
\def\aspp{Astroparticle Phys.}
\def\physrep{Physics Reports}

\def\sastl{Soviet Astron. Letters (Tr. Pisma Astr. Zh.)}
\def\azh{Astron. Zh.}
\def\arep{Astronomy Reports}
\def\astpbull{Astrofizicheskij bulletin}
\def\apbul{Astrophys. Bull.}


\def\jetp{ZhETF}
\def\jetpl{ZhETF Letters}

\def\apspr{Astrophysics and Space Physics Reviews}
\def\ssr{Space Science Reviews}

\def\afz{Astrophysics}

\def\pasp{PASP}
\def\apss{Ap\&SS}

\def\prl{Phys. Rev. Letters}
\def\prd{Phys. Rev. D}
\def\prc{Phys. Rev. C}
\def\plb{Physics Lett. B}
\def\jhep{J. of High Energy Phys.}
\def\jcap{J. of Cosmology and Astroparticle Phys.}
\def\aspp{Astroparticle Phys.}


\def\nuphs{Nuclear Phys.\ B Proc.\ Suppl.}
\def\phofl{Phys. of Fluids}
\def\ijmpa{International Journal of Modern Physics A}
\def\pnas{Publications of the National Academy of Science}

\def\spie{Proceedings of the SPIE}



\def\npb{Nuclear Phys.\ B}
\def\nuphs{Nuclear Phys.\ B Proc.\ Suppl.}
\def\phofl{Phys. of Fluids}

\def\gcnc#1#2{GCN Circ. #1 (#2)}
\def\iauc#1#2{IAU Circ. #1 (#2)}
\def\atel#1#2{The Astron. Circ.~\#~#1 (#2)}

\def\etal{\mbox{et al.}}

\renewenvironment{thebibliography}[1]{
%  \def\@biblabel#1{\relax}%
%  \stepcounter{NAT@ctr}

  \settowidth{\labelwidth}{#1.~}
  \renewcommand{\@biblabel}[1]{
    \stepcounter{NAT@ctr}
    \makebox[\labelwidth][r]{\arabic{NAT@ctr}.~}}
  \section{REFERENCES}
%  \medskip
  \list{
    \@biblabel{\@arabic\c@enumiv}}%
  {
    \@openbib@code
    \parindent=0pt \parskip=0pt \parsep=1pt
    \bibindent=0pt
    \small\interlinepenalty \@M  % AV
    \frenchspacing    % AV - to get right spacing after ``et al.''
    \hyphenpenalty=10000
%    \itemindent=-10pt   % îÅ ÔÁË ËÁË × ðáöÅ, ÎÏ ËÒÁÓÉ×ÅÅ
    \itemindent=-7pt   
%    \itemindent=0pt 
%    \settowidth{\labelwidth}{22.~}
%    \itemindent=-\labelwidth
    \itemsep=0pt                            %
    \listparindent=0pt                      %
%    \settowidth\labelwidth{\@biblabel{#1}}  %
    \labelsep=0pt                           %
%    \leftmargin=1.0em
    \leftmargin=24pt
    \advance\leftmargin\labelsep
%    \usecounter{enumiv}%
%    \let\p@enumiv\@empty
%    \renewcommand\theenumiv{\@arabic\c@enumiv}
  }%
  \sloppy\clubpenalty4000\widowpenalty4000%
  \sfcode`\.\@m}
{\def\@noitemerr{\@latex@warning{Empty `thebibliography' environment}}%
  \endlist}


\renewcommand\bibitem{%
  \@ifnextchar[{\@lbibitem}{%
    \global\NAT@stdbsttrue
    \stepcounter{NAT@ctr}\@lbibitem[\arabic{NAT@ctr}]}}
\def\@lbibitem[#1]#2{%
  \if\relax\@extra@b@citeb\relax\else
    \@ifundefined{br@#2\@extra@b@citeb}{}{%
     \@namedef{br@#2}{\@nameuse{br@#2\@extra@b@citeb}}}\fi
   \@ifundefined{b@#2\@extra@b@citeb}{\def\NAT@num{}}{\NAT@parse{#2}}%
   \item[\hfil\hyper@natanchorstart{#2\@extra@b@citeb}\@biblabel{\NAT@num}%
    \hyper@natanchorend]%
    \NAT@ifcmd#1(@)(@)\@nil{#2}}


% HEADINGS

\def\shortauthor#1{\gdef\@shortauthor{#1}}\shortauthor{}
\def\shorttitle#1{\gdef\@shorttitle{#1}}\shorttitle{}

\if@manuscript\else
 \RequirePackage{fancyhdr}
 \AtBeginDocument{\pagestyle{fancy}}

% Twoside
 \fancyhead{}
 \fancyhead[CE]{\@shortauthor}
 \fancyhead[LE]{\thepage}
 \fancyhead[RO]{\thepage}
 \fancyhead[CO]{\@shorttitle}
 \fancyfoot[CO]{}\fancyfoot[CE]{}
 \fancyfoot[CO]{}\fancyfoot[CE]{}
 \fancyfoot[RO]{\small ASTRONOMY LETTERS
   \quad Vol.~\@pazhvolume \quad  
%\textnumero\@pazhno 
No.~\@pazhno 
\quad \@pazhyear \quad}
 \fancyfoot[LE]{\small\quad ASTRONOMY LETTERS
   \quad Vol.~\@pazhvolume \quad  
%\textnumero\@pazhno 
No.~\@pazhno 
\quad \@pazhyear}
 \renewcommand{\headrulewidth}{0pt}
\fi

%
%\RequirePackage{indentfirst}
% \RequirePackage{amsmath,amssymb}
% 
% \renewcommand\normalsize{%
%    \@setfontsize\normalsize\@xipt{13.6}%
%    \abovedisplayskip 11\p@ \@plus3\p@ \@minus6\p@
%    \abovedisplayshortskip \z@ \@plus3\p@
%    \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@
%    \belowdisplayskip \abovedisplayskip
%    \let\@listi\@listI}
% \normalsize
% \renewcommand\small{%
%    \@setfontsize\small\@xpt\@xiipt
%    \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
%    \abovedisplayshortskip \z@ \@plus3\p@
%    \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
%    \def\@listi{\leftmargin\leftmargini
%                \topsep 6\p@ \@plus2\p@ \@minus2\p@
%                \parsep 3\p@ \@plus2\p@ \@minus\p@
%                \itemsep \parsep}%
%    \belowdisplayskip \abovedisplayskip
% }
% \renewcommand\footnotesize{%
%    \@setfontsize\footnotesize\@ixpt{11}%
%    \abovedisplayskip 8\p@ \@plus2\p@ \@minus4\p@
%    \abovedisplayshortskip \z@ \@plus\p@
%    \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
%    \def\@listi{\leftmargin\leftmargini
%                \topsep 4\p@ \@plus2\p@ \@minus2\p@
%                \parsep 2\p@ \@plus\p@ \@minus\p@
%                \itemsep \parsep}%
%    \belowdisplayskip \abovedisplayskip
% }
% \renewcommand\scriptsize{\@setfontsize\scriptsize\@viiipt{9.5}}
% \renewcommand\tiny{\@setfontsize\tiny\@vipt\@viipt}
% \renewcommand\large{\@setfontsize\large\@xiipt{14}}
% \renewcommand\Large{\@setfontsize\Large\@xivpt{18}}
% \renewcommand\LARGE{\@setfontsize\LARGE\@xviipt{22}}
% \renewcommand\huge{\@setfontsize\huge\@xxpt{25}}
% \renewcommand\Huge{\@setfontsize\Huge\@xxvpt{30}}
% 


% Symbols
\def\le{\leqslant}
\def\ge{\geqslant}

\newcommand\arcdeg{\mbox{$^\circ$}}
\newcommand\arcmin{\mbox{$^\prime$}}
\newcommand\arcsec{\mbox{$^{\prime\prime}$}}%

\DeclareRobustCommand{\N}{\textnumero}
%\DeclareTextSymbol{\textnumero}{OT2}{125}
\DeclareTextSymbol{\textnumero}{OT1}{185}
%\DeclareTextSymbol{\textnumero}{TS1}{155}

% defs from aa.cls:

\def\sun{\hbox{$\odot$}}
\def\la{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr<\cr\sim\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr
<\cr\sim\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr
<\cr\sim\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
<\cr\sim\cr}}}}}
\def\ga{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr>\cr\sim\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr
>\cr\sim\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr
>\cr\sim\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
>\cr\sim\cr}}}}}
\def\degr{\hbox{$^\circ$}}
%\def\arcmin{\hbox{$^\prime$}}
%\def\arcsec{\hbox{$^{\prime\prime}$}}
\def\utw{\smash{\rlap{\lower5pt\hbox{$\sim$}}}}
\def\udtw{\smash{\rlap{\lower6pt\hbox{$\approx$}}}}
\def\fd{\hbox{$.\!\!^{\rm d}$}}
\def\fh{\hbox{$.\!\!^{\rm h}$}}
\def\fm{\hbox{$.\!\!^{\rm m}$}}
\def\fs{\hbox{$.\!\!^{\rm s}$}}
\def\fdg{\hbox{$.\!\!^\circ$}}
\def\farcm{\hbox{$.\mkern-4mu^\prime$}}
\def\farcs{\hbox{$.\!\!^{\prime\prime}$}}
\def\fp{\hbox{$.\!\!^{\scriptscriptstyle\rm p}$}}
\def\cor{\mathrel{\mathchoice {\hbox{$\widehat=$}}{\hbox{$\widehat=$}}
{\hbox{$\scriptstyle\hat=$}}
{\hbox{$\scriptscriptstyle\hat=$}}}}
\def\sol{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr\sim\cr<\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr\sim\cr
<\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr\sim\cr
<\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
\sim\cr<\cr}}}}}
\def\sog{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr\sim\cr>\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr\sim\cr
>\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr
\sim\cr>\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
\sim\cr>\cr}}}}}
\def\lse{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr<\cr\simeq\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr
<\cr\simeq\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr
<\cr\simeq\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
<\cr\simeq\cr}}}}}
\def\gse{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr>\cr\simeq\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr
>\cr\simeq\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr
>\cr\simeq\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
>\cr\simeq\cr}}}}}
\def\grole{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr>\cr\noalign{\vskip-1.5pt}<\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr
>\cr\noalign{\vskip-1.5pt}<\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr
>\cr\noalign{\vskip-1pt}<\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
>\cr\noalign{\vskip-0.5pt}<\cr}}}}}
\def\leogr{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr<\cr\noalign{\vskip-1.5pt}>\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr
<\cr\noalign{\vskip-1.5pt}>\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr
<\cr\noalign{\vskip-1pt}>\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
<\cr\noalign{\vskip-0.5pt}>\cr}}}}}
\def\loa{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr<\cr\approx\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr
<\cr\approx\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr
<\cr\approx\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
<\cr\approx\cr}}}}}
\def\goa{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr>\cr\approx\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr
>\cr\approx\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr
>\cr\approx\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
>\cr\approx\cr}}}}}
\def\diameter{{\ifmmode\mathchoice
{\ooalign{\hfil\hbox{$\displaystyle/$}\hfil\crcr
{\hbox{$\displaystyle\mathchar"20D$}}}}
{\ooalign{\hfil\hbox{$\textstyle/$}\hfil\crcr
{\hbox{$\textstyle\mathchar"20D$}}}}
{\ooalign{\hfil\hbox{$\scriptstyle/$}\hfil\crcr
{\hbox{$\scriptstyle\mathchar"20D$}}}}
{\ooalign{\hfil\hbox{$\scriptscriptstyle/$}\hfil\crcr
{\hbox{$\scriptscriptstyle\mathchar"20D$}}}}
\else{\ooalign{\hfil/\hfil\crcr\mathhexbox20D}}%
\fi}}

\def\getsto{\mathrel{\mathchoice {\vcenter{\offinterlineskip
\halign{\hfil
$\displaystyle##$\hfil\cr\gets\cr\to\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr\gets
\cr\to\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr\gets
\cr\to\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
\gets\cr\to\cr}}}}}
\def\lid{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr<\cr\noalign{\vskip1.2pt}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr<\cr
\noalign{\vskip1.2pt}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr<\cr
\noalign{\vskip1pt}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
<\cr
\noalign{\vskip0.9pt}=\cr}}}}}
\def\gid{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr>\cr\noalign{\vskip1.2pt}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr>\cr
\noalign{\vskip1.2pt}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr>\cr
\noalign{\vskip1pt}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
>\cr
\noalign{\vskip0.9pt}=\cr}}}}}
\def\bbbr{{\rm I\!R}} %reelle Zahlen
\def\bbbm{{\rm I\!M}}
\def\bbbn{{\rm I\!N}} %natuerliche Zahlen
\def\bbbf{{\rm I\!F}}
\def\bbbh{{\rm I\!H}}
\def\bbbk{{\rm I\!K}}
\def\bbbp{{\rm I\!P}}
\def\bbbone{{\mathchoice {\rm 1\mskip-4mu l} {\rm 1\mskip-4mu l}
{\rm 1\mskip-4.5mu l} {\rm 1\mskip-5mu l}}}
\def\bbbc{{\mathchoice {\setbox0=\hbox{$\displaystyle\rm C$}\hbox{\hbox
to0pt{\kern0.4\wd0\vrule height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\textstyle\rm C$}\hbox{\hbox
to0pt{\kern0.4\wd0\vrule height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptstyle\rm C$}\hbox{\hbox
to0pt{\kern0.4\wd0\vrule height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptscriptstyle\rm C$}\hbox{\hbox
to0pt{\kern0.4\wd0\vrule height0.9\ht0\hss}\box0}}}}
\def\bbbq{{\mathchoice {\setbox0=\hbox{$\displaystyle\rm
Q$}\hbox{\raise
0.15\ht0\hbox to0pt{\kern0.4\wd0\vrule height0.8\ht0\hss}\box0}}
{\setbox0=\hbox{$\textstyle\rm Q$}\hbox{\raise
0.15\ht0\hbox to0pt{\kern0.4\wd0\vrule height0.8\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptstyle\rm Q$}\hbox{\raise
0.15\ht0\hbox to0pt{\kern0.4\wd0\vrule height0.7\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptscriptstyle\rm Q$}\hbox{\raise
0.15\ht0\hbox to0pt{\kern0.4\wd0\vrule height0.7\ht0\hss}\box0}}}}
\def\bbbt{{\mathchoice {\setbox0=\hbox{$\displaystyle\rm
T$}\hbox{\hbox to0pt{\kern0.3\wd0\vrule height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\textstyle\rm T$}\hbox{\hbox
to0pt{\kern0.3\wd0\vrule height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptstyle\rm T$}\hbox{\hbox
to0pt{\kern0.3\wd0\vrule height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptscriptstyle\rm T$}\hbox{\hbox
to0pt{\kern0.3\wd0\vrule height0.9\ht0\hss}\box0}}}}
\def\bbbs{{\mathchoice
{\setbox0=\hbox{$\displaystyle     \rm S$}\hbox{\raise0.5\ht0\hbox
to0pt{\kern0.35\wd0\vrule height0.45\ht0\hss}\hbox
to0pt{\kern0.55\wd0\vrule height0.5\ht0\hss}\box0}}
{\setbox0=\hbox{$\textstyle        \rm S$}\hbox{\raise0.5\ht0\hbox
to0pt{\kern0.35\wd0\vrule height0.45\ht0\hss}\hbox
to0pt{\kern0.55\wd0\vrule height0.5\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptstyle      \rm S$}\hbox{\raise0.5\ht0\hbox
to0pt{\kern0.35\wd0\vrule height0.45\ht0\hss}\raise0.05\ht0\hbox
to0pt{\kern0.5\wd0\vrule height0.45\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptscriptstyle\rm S$}\hbox{\raise0.5\ht0\hbox
to0pt{\kern0.4\wd0\vrule height0.45\ht0\hss}\raise0.05\ht0\hbox
to0pt{\kern0.55\wd0\vrule height0.45\ht0\hss}\box0}}}}
\def\bbbz{{\mathchoice {\hbox{$\sf\textstyle Z\kern-0.4em Z$}}
{\hbox{$\sf\textstyle Z\kern-0.4em Z$}}
{\hbox{$\sf\scriptstyle Z\kern-0.3em Z$}}
{\hbox{$\sf\scriptscriptstyle Z\kern-0.2em Z$}}}}
\def\ts{\thinspace}

