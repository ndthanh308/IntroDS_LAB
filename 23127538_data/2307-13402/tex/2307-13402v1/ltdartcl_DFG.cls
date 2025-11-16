\ProvidesClass{ltdartcl_dfg}[2021/07/21]
% Based on scrartcl, hence the name ltdartcl
\LoadClassWithOptions{scrartcl}

\usepackage{geometry}
\geometry{left=1.8cm,right=1.8cm,top=2.2cm,bottom=3cm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% misc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\setlength{\parindent}{0mm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Math
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage{amsmath,amsfonts,latexsym}
\usepackage{amssymb}
% bold in math
%\usepackage{bm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graphics and Colors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Graphics
\usepackage{graphicx}
 
% Colors
\usepackage[rgb,override]{xcolor}
\definecolor{caltechgray}{gray}{0.65}
\newcommand{\caltechgray}[1]{\textcolor{caltechgray}{#1}}  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Header and Footer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage{fancyhdr}
\pagestyle{fancy}
\setlength{\headheight}{33pt}
% Logos in the header
\fancyhead[L]{\includegraphics[height=1cm]{Logo_Tech-Fak_DinA4_4C}}
\fancyhead[R]{\includegraphics[height=1cm]{LTD_Logo_mit_prof}}
%\fancyhead[R]{\includegraphics[height=1cm]{ltdltdlogo}}
\fancyfoot[R]{\footnotesize {\sffamily {\upshape \caltechgray{Stand: \today}}}}

% initial header and footer
\fancyfoot[L]{}
\fancyfoot[C]{}
\fancyhead[C]{}

% commands for changing header and footer in the document
\newcommand{\ltddocname}[1]{
% document name
\fancyfoot[L]{\footnotesize {\sffamily {\upshape \caltechgray{#1}}}}
}
\newcommand{\ltdcreator}[2]{
% creator name
\fancyhead[C]{\footnotesize {\sffamily {\upshape \caltechgray{#1}}}}
% Creator E-Mail
\fancyfoot[C]{\footnotesize {\sffamily {\upshape \caltechgray{#2}}}}
}
% document name and creator
\newcommand{\ltdsetup}[3]{
\ltddocname{#1}
\ltdcreator{#2}{#3}
}




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% separating lines for footer and header
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\renewcommand{\headrule}{{\color{caltechgray}\hrule width\headwidth height\headrulewidth \vskip-\headrulewidth}}

\renewcommand{\footrule}{{\color{caltechgray}\vskip-\footruleskip\vskip-\footrulewidth\hrule width\headwidth height\footrulewidth\vskip\footruleskip}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LTD-Headline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\ltdheadline}[2]{
\begin{center}
{\sffamily \large{\textbf{#1}}}\\[.2cm]

{\sffamily #2}
\end{center}
}
