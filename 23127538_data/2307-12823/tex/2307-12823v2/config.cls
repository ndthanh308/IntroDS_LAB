\ProvidesClass{config}
\LoadClass[aps,pra,twocolumn,amsmath,amssymb,nofootinbib,showpacs,superscriptaddress]{revtex4-1}

\usepackage[english]{babel}
\usepackage{latexsym}
\usepackage{graphics}
\usepackage{graphicx}
\usepackage{epsfig}
\usepackage{color}
\usepackage{bm}
\usepackage{amsmath}
\usepackage{bbm}
\usepackage{bbold}
\usepackage{dsfont}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{dcolumn}
\usepackage{float}
\usepackage{hyperref}
\usepackage{color}
\usepackage{epstopdf}
\usepackage{cleveref}
\usepackage[svgnames]{xcolor}
\usepackage{dsfont}
\hypersetup{hidelinks,colorlinks=true,allcolors=DarkBlue}
\usepackage{subcaption}

\graphicspath{{imgs/}}

\newcommand{\der}[1]{\dfrac{\partial}{\partial #1}}
\newcommand{\dder}[2]{\dfrac{\partial #1}{\partial #2}}
\newcommand{\Sum}{\displaystyle\sum\limits}
\newcommand{\Prod}{\displaystyle\prod\limits}
\newcommand{\Int}{\displaystyle\int\limits}
\newcommand{\Lim}{\displaystyle\lim\limits}
\newcommand{\Max}{\max\limits}
\newcommand{\Min}{\min\limits}
\newcommand{\RR}{\mathbb{R}}
\newcommand{\CC}{\mathbb{C}}
\newcommand{\id}{\mathds{1}}
\newcommand{\hil}{\mathcal{H}}
\newcommand{\denset}{\mathcal{S}(\hil)}
\newcommand{\choi}{C_{\mathcal{E}}}
\newcommand{\povm}{\{E_i\}_{i=1}^P}
\newcommand{\mlerho}{\hat\rho^{\text{MLE}}}
\newcommand{\linrho}{\hat\rho^{\text{LS}}}
\newcommand{\inpbasis}{\{\rho_i\}_{i=1}^{M}}
\newcommand{\outbasis}{\{\mathcal{E}(\rho_i)\}_{i=1}^{M}}
\newcommand{\mlechoi}{\hat C_{\mathcal{E}}^{\text{MLE}}}
\newcommand{\linchoi}{\hat C_{\mathcal{E}}^{\text{LS}}}
\newcommand{\cptp}{\mathcal{CPTP}}
\newcommand{\cp}{\mathcal{CP}}
\newcommand{\tp}{\mathcal{TP}}

\newcommand{\ketbra}[1]{|#1\rangle\langle#1|}
\newcommand{\ket}[1]{|#1\rangle}
\newcommand{\bra}[1]{\langle#1|}

\renewcommand{\le}{\leqslant}
\renewcommand{\ge}{\geqslant}
\renewcommand{\Pr}[1]{\text{Pr}\left[#1\right]}

\DeclareMathOperator{\Tr}{Tr}
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator{\Mult}{Mult}