% This is an official LaTeX class for writing articles for the RADIOENGINEERG Journal.
% version 0.2
% Created in 2009 by Martin Slanina, last changes 01/2018 by Jiri Milos
%
% Based on a work by:
%		Jakub Stastny 	(2001)
% 		Miroslav Strupl (2003)
% 		Josef Dobes 		(2005)

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{radioeng_modified}[2009/01/26 Radioengineering Journal Class]
\LoadClass[a4paper, twoside]{article}
\RequirePackage[T1]{fontenc}
\RequirePackage{newtxtext,newtxmath}
\RequirePackage{fancyhdr}
\RequirePackage{multicol}
\RequirePackage{caption}
\RequirePackage{atbeginend}
\RequirePackage{tweaklist}
\RequirePackage{ifthen}
\RequirePackage{numprint}

\newcounter{firstpage}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define math typeset commands
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\hyphenation{an-a-lys-is}
\DeclareMathSymbol{\imag}{\mathord}{operators}{`i} % imaginary unit i
\DeclareMathSymbol{\jmag}{\mathord}{operators}{`j} % imaginary unit j

\DeclareMathSymbol{\diff}{\mathord}{operators}{`d} % derivate with respect to, e.g. dx
\DeclareMathSymbol{\eul}{\mathord}{operators}{`e} % mathematical constant e

\def\upmicro{\mbox{\textmu}} %% Definice symbolu "micro" in uprigth 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define fonts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\headerfont}{\fontsize{9pt}{1}\sffamily\selectfont}
\newcommand{\tablefont}{\fontsize{8pt}{10pt}\selectfont}
\newcommand{\titlefont}{\fontseries{b}\fontshape{n}\fontsize{20pt}{24pt}\selectfont}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define non-standard counters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{appendix}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define section and subsection headline format
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\rengSection}[1]{\refstepcounter{section}
        \vspace{3em}\vspace{-3mm}
        \noindent{\parbox[t]{0.75cm}{\Large\bfseries
        \arabic{section}.}\parbox[t]{77mm}
        {\Large\bfseries #1}}
        \vspace{-4mm}
        \nopagebreak[5]}
        

\newcommand{\rengSubsection}[1]{\refstepcounter{subsection}
        \vspace{2em}\vspace{-3mm}
        \noindent{\parbox[t]{6.6mm}{\normalfont
        \fontsize{12pt}{1}\fontseries{b}\selectfont
        \arabic{section}.\arabic{subsection}}
        \parbox[t]{77mm}{\normalfont
        \fontsize{12pt}{1}\fontseries{b}\selectfont
        #1}}
        \vspace{-4mm}
        \nopagebreak[5]}

\newcommand{\rengSubsubsection}[1]{\refstepcounter{subsubsection}
        \vspace{1em}\vspace{-3mm}
        \noindent{\parbox[t]{6.6mm}{\normalfont
        \fontsize{10pt}{1}\fontseries{b}\selectfont
        \arabic{section}.\arabic{subsection}.\arabic{subsubsection}}
        \parbox[t]{77mm}{\normalfont
        \fontsize{10pt}{1}\fontseries{b}\selectfont
        #1}}
        \vspace*{-1.5em}
        \vspace{-4mm}
        \nopagebreak[5]}
        
\newcommand{\rengAppendix}[1]{\refstepcounter{appendix}
 \setcounter{equation}{0}
\renewcommand\theequation{\Alph{appendix}\arabic{equation}}
        \vspace{3em}\vspace{-3mm}
        \noindent{\parbox[t]{0.75cm}{\Large\bfseries
        }\parbox[t]{77mm}
        {\Large\bfseries Appendix \Alph{appendix}: #1}}
        \vspace{-4mm}
        \nopagebreak[5]}
      
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define page size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\setlength{\hoffset}{-28mm} 	% Original value
\setlength{\voffset}{-21.5mm} % Original value
\addtolength{\voffset}{2mm}
\setlength{\oddsidemargin}{22.6mm} % +2.6mm, checked using the pages overlay
\setlength{\evensidemargin}{17.6mm} % Consequence of above, odd - even differ with 5mm
\setlength{\topmargin}{8.5mm} % Shortened of 1.5mm, checked using the pages overlay
\setlength{\textwidth}{175mm} % Measured
\setlength{\textheight}{241mm} % Decreased of \topmargin (10mm+1.5mm+0.5mm header shift)
\setlength{\headsep}{10mm} % Measured
\setlength{\parindent}{7.5mm} % Measured
\setlength{\parskip}{2mm} % (\addtolength{\parskip}{6pt}), 2mm measured

\setlength{\columnsep}{6mm}

\arrayrulewidth = 0.5 pt

\pagestyle{fancy}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Floating object spacing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\BeforeBegin{figure}{\vspace{-0.2cm}}
\AfterEnd{figure}{\vspace{-0.3cm}}
\BeforeBegin{table}{\vspace{-0.3cm}}
\AfterEnd{table}{\vspace{-0.3cm}}
\BeforeBegin{equation}{\vspace{-0.1cm}}
\AfterEnd{equation}{\vspace{-0.45cm}}

\renewcommand{\itemhook}{
  \setlength{\topsep}{0cm}%
  \setlength{\leftmargin}{0.75cm}
  %\setlength{\itemsep}{0cm}
  }
\renewcommand{\enumhook}{
  \setlength{\topsep}{0cm}%
  \setlength{\leftmargin}{0.75cm}
  %\setlength{\itemsep}{0cm}
  }



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Page headers, title, authors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% the header of the page
% usage: \rengHeaded{vol}{no}{date}{paper}
\newcommand{\rengHeader}[4]
{
\setcounter{page}{\value{firstpage}}
%\fancyhead[LO]{\headerfont RADIOENGINEERING, VOL. #1, NO. #2, #3\\[0.45mm]}
\fancyhead[CO]{}
\fancyhead[RO]{\headerfont \thepage\\ [1.1mm]}
\fancyhead[LE]{\headerfont \thepage\\ [1.1mm]}
\fancyhead[CE]{}
\fancyhead[RE]{\headerfont #4\\ [0.45mm]}
}
\nplpadding[0]{4}
\npthousandsep{}
%\fancyfoot[L]{\ifthenelse{\value{page}=\value{firstpage}}{\headerfont DOI: 10.13164/re.2021.\numprint{\thefirstpage}}{}}
\fancyfoot[C]{}
%\fancyfoot[R]{\ifthenelse{\value{page}=\value{firstpage}}{\headerfont\area}{}}

%\lfoot{DOI.13164/re.2015.0001}
%\rfoot{}
%\cfoot{}

% the title of the paper
% usage: \rengTitle{title text}
\newcommand{\rengTitle}[1]
{
	\begin{center}
		\vspace*{-6.5mm}
		\titlefont{#1}
	\end{center}
}

\newcommand{\rengNames}[1]
{
	\begin{center} 
%  \vspace{2em}
	\vspace{0.07cm}
  {\fontsize{12pt}{1}\fontshape{it}\selectfont
  #1\\
  }
  \end{center}
}

\newcommand{\rengAffil}[1]
{
	\begin{center}
%  \vspace{1em}
	\vspace{-0.12cm}
  #1\\
	\end{center}
}

\newcommand{\rengMail}[1]
{
	\begin{center}
	\vspace{-0.12cm}
	#1\\
	\end{center}
}

\newcommand{\rengReceived}[2]
{
	\begin{center}
	\vspace{-0.14cm}
	\footnotesize{Submitted #1~/~Accepted #2}
	\vspace{0.42cm}
	\end{center}
}



\newenvironment{rengAbstract}
{\noindent\textbf{Abstract. }\addtolength{\parindent}{-7.5mm}\itshape}
{\normalfont}

% the keywords
% usage: \rengKeywords{list of keywords}
\newcommand{\rengKeywords}[1]
{
	\vspace{3.5em}
	\vspace{-0.2cm}
	\noindent{\bf{\Large Keywords\\}}
	
	\vspace{-0.35cm}
	%\hspace*{0.00cm}
	\parbox[t]{22em}{#1}
	\vspace{0.0 cm}
}

\newcommand{\rengAck}
{ 	
	\vspace{3em}
	\noindent{\Large\bfseries Acknowledgments}
}

\newenvironment{tablehere}
  {\def\@captype{table}}
  {}

\renewenvironment{table}
  {\def\@captype{table}}
  {}
  
\renewenvironment{figure}
  {\def\@captype{figure}}
  {}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define figure and table title format
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\DeclareCaptionLabelFormat{fig}{\hspace{2.5mm}Fig. #2}
\DeclareCaptionLabelFormat{tab}{\hspace{2.5mm}Tab. #2}
\DeclareCaptionLabelSeparator{dot}{. }
%\DeclareCaptionFormat{reng}{}
\captionsetup{
format=hang, 
labelsep=dot,
font=footnotesize, 
labelfont=bf, 
justification=justified, 
singlelinecheck=false, 
margin=5mm,
position=bottom,
aboveskip=2mm
}
\captionsetup[figure]{position=below}

\newcommand{\tcaption}[1]{
\captionsetup{labelformat=tab}
\caption{#1}
}

\newcommand{\fcaption}[1]{
\captionsetup{labelformat=fig}
\caption{#1}
}

% Font size of table
%\let\oldtabular\tabular
%\renewcommand{\tabular}{\footnotesize\oldtabular}
\setlength\arrayrulewidth{1pt}