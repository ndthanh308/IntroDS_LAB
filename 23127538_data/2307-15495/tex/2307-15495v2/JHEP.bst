\documentclass[a4paper,12pt]{article}
%\pdfoutput=1 % if your are submitting a pdflatex (i.e. if you have
             % images in pdf, png or jpg format)

\PassOptionsToPackage{table}{xcolor}
\usepackage{jheppub} % for details on the use of the package, please
                     % see the JHEP-author-manual

\usepackage[T1]{fontenc} % if needed




\parskip 3 pt
 %\marginparwidth 0pt
\oddsidemargin  1.7cm
\evensidemargin 0pt
 %\marginparsep 0pt
\topmargin   1mm
\textwidth   17cm
\textheight  25.8cm

\renewcommand{\baselinestretch}{1.1}
\newcommand{\overbar}[1]{\mkern 1.5mu\overline{\mkern-1.5mu#1\mkern-1.5mu}\mkern 1.5mu}
\renewcommand{\theequation}{\thesection.\arabic{equation}}


\usepackage[all]{xy}
%%%%%%%%%%%%%%%%%%%%%%%%
%\usetikzlibrary{decorations.pathmorphing}
\usepackage[percent]{overpic}
\usepackage{slashed}
\usepackage{wrapfig}
\usepackage{tabu}
\usepackage{diagbox}
\usepackage{mathrsfs,amsmath,amssymb,amsthm,amsfonts,tikz,graphicx,accents,hyperref, color,bbold}
\usepackage{dsfont,epiolmec, latexsym, stmaryrd, comment}
\usepackage{slashed,ccaption}
\usepackage{mathrsfs, calligra}
\usepackage{leftidx}
\usepackage{import}
\usepackage{multirow}
\usepackage{amsfonts}
\usepackage{pifont}
\usepackage{tabularx}
\usepackage[utf8]{inputenc}
\usetikzlibrary{intersections,calc}
\usepackage{tikz-3dplot}
\usepackage{ifthen}
\usetikzlibrary{arrows}
%\usepackage{amsmath}
\usepackage{xcolor}
%\usepackage{geometry}

\usepackage{caption} 
\DeclareMathOperator{\extdm}{d}
\newcommand{\extd}{\extdm \!}


%\usepackage[table]{xcolor} 
\usepackage{array}
%

%%%%%%%%%%%%%%%%%%%%%%


\usepackage[percent]{overpic}
\usepackage{wrapfig}
\usepackage{bbm}
\usepackage{tabu}
\usepackage{slashed}
\usepackage{fancyhdr} % package for changing Headings style
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{diagbox}
%\usepackage{slashbox}
%\usepackage{pict2e}
%\usepackage[demo]{graphicx}
%\usepackage{caption}
%\usepackage{subcaption}

\hypersetup{ linktoc=all,
    colorlinks, linkcolor={blue}, %{vividviolet},
    citecolor={red}, urlcolor={darkpink}
}
\newcommand{\changeurlcolor}[1]{\hypersetup{urlcolor=#1}}
\graphicspath{{Images/}}

\definecolor{light gray}{RGB}{220,220,220}
\definecolor{dark purple}{RGB}{108,0,217}
\definecolor{pink}{RGB}{190,20,100}
\definecolor{orang}{RGB}{193,63,0}
\definecolor{green}{RGB}{11,98,17}
\definecolor{darkpink}{RGB}{153,0,76}
\definecolor{bluegreen}{RGB}{0,102,102}
\definecolor{greenlagan}{RGB}{0,102,0}
\definecolor{redgreen}{RGB}{102,102,0}
\definecolor{Redgreen}{RGB}{153,76,0}
\definecolor{vividviolet}{rgb}{0.62, 0.0, 1.0}
\definecolor{amaranth}{rgb}{0.9, 0.17, 0.31}
\definecolor{palatinateblue}{rgb}{0.15, 0.23, 0.89}
\definecolor{brightpink}{rgb}{1.0, 0.0, 0.5}
\definecolor{cornflowerblue}{rgb}{0.39, 0.58, 0.93}
\definecolor{deepcarminepink}{rgb}{0.94, 0.19, 0.22}
\definecolor{radicalred}{rgb}{1.0, 0.21, 0.37}
%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows,chains,shapes,matrix,positioning,scopes}
%\usetikzlibrary{arrows,shapes,positioning}
\usetikzlibrary{decorations.markings}
\tikzstyle arrowstyle=[scale=1]
\tikzstyle directed=[postaction={decorate,decoration={markings,
    mark=at position .65 with {\arrow[arrowstyle]{stealth}}}}]
\tikzstyle reverse directed=[postaction={decorate,decoration={markings,
    mark=at position .65 with {\arrowreversed[arrowstyle]{stealth};}}}]
\usepackage{import}
\usepackage{accents}
\usepackage{mathrsfs,amsmath,amssymb,slashed}
\usepackage{multirow,multicol}
\usepackage{enumitem}
%%%%%%%%%%%%%%%%%%%%%%%%
%\usetikzlibrary{decorations.pathmorphing}
\usepackage[percent]{overpic}
\usepackage{slashed}
\newcommand{\pii}{\textbf{p}}
\newcommand{\x}{\text{x}}
\usepackage{wrapfig}
\usepackage{tabu}
\usepackage{diagbox}
\usepackage{comment}
\usepackage{tikz}
\usepackage{mathrsfs,amsmath,amssymb,amsthm,amsfonts,graphicx,accents,hyperref,color}
\usepackage{leftidx}
\usepackage{import}
%\usetikzlibrary{snakes}
\usetikzlibrary{decorations.pathmorphing}
%
\DeclareFontFamily{OT1}{rsfs}{}

\DeclareFontShape{OT1}{rsfs}{m}{n}{ <-7> rsfs5 <7-10> rsfs7 <10->rsfs10}{} 

\DeclareMathAlphabet{\mycal}{OT1}{rsfs}{m}{n}
%
\def\mathbi#1{\textbf{\em #1}}
%


%\definecolor{darkpink}{RGB}{153,0,76}
\newcommand{\tcb}{\textcolor{blue}}
\newcommand{\tcg}{\textcolor{green}}
\newcommand{\tcr}{\textcolor{green}}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand\hnote[1]{\textcolor{blue}{\bf [H:\,#1]}}
\newcommand\vnote[1]{\textcolor{blue}{\bf [V:\,#1]}}
\newcommand\hrnote[1]{\textcolor{orange}{\bf [HR:\,#1]}}
\newcommand\lnote[1]{\textcolor{darkpink}{\bf [L:\,#1]}}
%%%%%%%%%%
\title{{\LARGE{Results On Open Quantum Systems}}}
%\author[a]{S. Shajidul Haque}
%\author[a,b]{, Ghadir Jafari}
%\author[c]{, Bret Underwood}
%
%
%
%\affiliation{\it $^a$ Department of Mathematics and Applied Mathematics,
%	University of Cape Town, South Africa
%}
%
%\affiliation{\it $^b$ School of Particles and Accelerators, Institute for Research in Fundamental
%	Sciences (IPM),\\ P.O.Box 19395-5531, Tehran, Iran}
%
%\affiliation{\it $^c$ Department of Physics, Pacific Lutheran University, Tacoma, WA 98447
%}

%\emailAdd{shajid.haque@uct.ac.za }
%\emailAdd{ghjafari@ipm.ir}
%\emailAdd{bret.underwood@plu.edu}
%


\abstract{}    
\begin{document}
\maketitle
%\newpage
%%%%%%%n example where the boundary conditions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Introduction}
\section{SFF for non-gaussian models}
For the Sextic Potential:
\begin{equation}
	V(M)=\frac{1}{2} M^2+\frac{g}{N} M^4+\frac{h}{N^2} M^6
\end{equation}
The SFF as a function of time  will be as  fig: (\ref{fig:plt1})
\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{plt1}
	\caption{}
	\label{fig:plt1}
\end{figure}
And the averaging will be like fig: (\ref{fig:plt2})
\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{plt2}
	\caption{}
	\label{fig:plt2}
\end{figure}
we see that the effect of non-gaussianity is to change the minimum time and  ramp slope, but the late time behavior remains unaffected. 

\section{Open Quantum Systems}
The effect of environment on the SFF for the gaussian model is according to fig (\ref{fig:plt3})
\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{plt3}
	\caption{}
	\label{fig:plt3}
\end{figure}
Where $F_t$ is evaluated by 
\begin{equation}
	F_t=\frac{1}{2 \sqrt{\pi \gamma t}} \int_{-\infty}^{+\infty} d \tau e^{-\left(\frac{\tau-2 t}{2 \sqrt{\gamma t}}\right)^2} g_\beta(\tau)
\end{equation}
The averaging behavior is given by fig (\ref{fig:plt4})
\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{plt4}
	\caption{}
	\label{fig:plt4}
\end{figure}


Now,  let us compare the effect of non-gaussianity for different open systems. According to fig (\ref{fig:plt5}), the effect of environment for the non-gaussian model is also that it just decreases the length of the ramp. 
\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{plt5}
	\caption{}
	\label{fig:plt5}
\end{figure}
It will be more evident in the case of averaging  According to fig (\ref{fig:plt6}),
\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{plt6}
	\caption{}
	\label{fig:plt6}
\end{figure}
\section{Complexity}
The relation with complexity is going by computing the moments from the survival amplitude
\begin{equation}
	S(t)=\left\langle\psi_{\beta+2 i t} \mid \psi_\beta\right\rangle=\frac{Z_{\beta-i t}}{Z_\beta}
\end{equation}
which has the relation $SFF=S^2$ with the spectral form factor. The moments are given by:
\begin{equation}
	\left.\mu_n \equiv \frac{d^n}{d t^n} S(t)\right|_{t=0}=\frac{1}{Z_\beta} \operatorname{Tr}\left(e^{-\beta H}(i H)^n\right)
\end{equation}
From the moments $\mu_n $ we can calculate the Lanczos coefficients $a_n,b_n$ by using a  Markov chain. Then using the  Lanczos coefficients, we will find the wave functions $\psi_n(t)$ by: 
\begin{equation}
	i \partial_t \psi_n(t)=a_n \psi_n(t)+b_{n+1} \psi_{n+1}(t)+b_n \psi_{n-1}(t)
\end{equation}
and then the complexity is given by:
\begin{equation}
	C(t)=C_{\mathcal{K}}(t)=\sum_n n\left|\psi_n(t)\right|^2
\end{equation}
the moments grow with $n$  According to fig (\ref{fig:plt7}) for different systems. 
\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{plt7}
	\caption{}
	\label{fig:plt7}
\end{figure}
\section*{Acknowledgement}
%\nocite{*}
\bibliographystyle{utcaps}

%\printbibliography

\bibliography{reference}


\end{document}





