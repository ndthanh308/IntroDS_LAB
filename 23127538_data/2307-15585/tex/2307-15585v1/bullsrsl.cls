%%
%% This is file `bullsrsl.cls'.
%%
%% IMPORTANT NOTICE:
%% 
%% This program can be redistributed and/or modified under the terms
%% of the LaTeX Project Public License Distributed from CTAN
%% archives in directory macros/latex/base/lppl.txt; either
%% version 1 of the License, or any later version.
%% 
%% =============================================
%% Copyright 2022 Guy Munhoven
%% University of Liege
%% Inst. of Astrophysics and Geophysics
%% 19c, Allee du 6-Aout
%% B-4000 Liege
%% Belgium
%% E-mail: guy.munhoven@uliege.be

% Require LaTeX from 2003/12/01 or newer for UTF8 support
\NeedsTeXFormat{LaTeX2e}[2003/12/01]
\ProvidesClass{bullsrsl}[2022/06/16 Bull. Soc. R. Sci. Liege Article class v0.9 (GM)]


% === Class options and required packages ===

% There will be a few alternatives to handle: we need the "ifthen" package
\RequirePackage{ifthen}

%% article.csl options not used here
% a4paper always used
\DeclareOption{a5paper}{\OptionNotUsed}
\DeclareOption{b5paper}{\OptionNotUsed}
\DeclareOption{letterpaper}{\OptionNotUsed}
\DeclareOption{legalpaper}{\OptionNotUsed}
\DeclareOption{executivepaper}{\OptionNotUsed}
\DeclareOption{landscape}{\OptionNotUsed}   % Never
\DeclareOption{10pt}{\OptionNotUsed}
\DeclareOption{11pt}{\OptionNotUsed}
% 12pt always used
\DeclareOption{oneside}{\OptionNotUsed}     % N/A
\DeclareOption{twoside}{\OptionNotUsed}     % N/A
\DeclareOption{titlepage}{\OptionNotUsed}   % Never
\DeclareOption{notitlepage}{\OptionNotUsed} % N/A
\DeclareOption{onecolumn}{\OptionNotUsed}   % N/A
\DeclareOption{twocolumn}{\OptionNotUsed}   % Never
\DeclareOption{fleqn}{\OptionNotUsed}       % Never

%% Class options
%  - 'manuscript' vs. 'final' (default)
%TBD% make these mutually exclusive (if both are given, make 'final' override)
\newboolean{BSRSL@manuscript}
\newboolean{BSRSL@final}
\DeclareOption{manuscript}{\setboolean{BSRSL@manuscript}{true}
                           \setboolean{BSRSL@final}{false}}
\DeclareOption{final}{\setboolean{BSRSL@manuscript}{false}
                           \setboolean{BSRSL@final}{true}}

%  - 'english' vs. 'french' (default)
%TBD% make these mutually exclusive (if both are given, make 'french' override)
\newboolean{BSRSL@english}
\newboolean{BSRSL@french}
\DeclareOption{french}{\setboolean{BSRSL@english}{false}
                       \setboolean{BSRSL@french}{true}}
\DeclareOption{english}{\setboolean{BSRSL@english}{true}
                        \setboolean{BSRSL@french}{false}}

%  - to be added: invited, regular, review, conference
%    (actually relevant only for the 'final' paper


% Pass specified options to article.cls
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

% Apply default options 'final' and 'french' and process them
\ExecuteOptions{final,french}
\ProcessOptions*

% Load underlying "article" class with 12pt and a4paper options.
\LoadClass[a4paper,12pt]{article}

% Require Times New Roman font
\RequirePackage{mathptmx}

% Require "hyperref" package
% Include 'hidelinks' option to avoid having coloured boxes or
% coloured text where there are links.
\RequirePackage[hidelinks]{hyperref}
% Also typeset URLs in regular instead of the default typewriter font.
\urlstyle{same}

% Require the lineno package for line numbering
% (activated only with the manuscript option).
\RequirePackage{lineno}


% Adapt page geometry via geometry package
\RequirePackage[left=2.5cm,right=2.5cm,top=2.5cm,bottom=1.25cm,includefoot]{geometry}


% Use the keyval package to process the \author[]{}{} optional argument list
% (keyval is part of the standard graphics collection in LaTeX). See
% https://tex.stackexchange.com/questions/34312/how-to-create-a-command-with-key-values/34314#34314
% for an example of usage.
\RequirePackage{keyval}


% === Required packages done ===



% === Auxiliary functions and definitions ===
% Function to carry out a one-stage only (not a full) expansion.
% (from https://tex.stackexchange.com/questions/116823/partially-expanding-a-command#comment259177_116825)
\newcommand{\expand@once}[1]{\unexpanded\expandafter{#1}}

% \and@et: parametric representation of the word "and" in English and French.
\ifthenelse{\boolean{BSRSL@french}}
   {\def\and@et{et}}{\def\and@et{and}}


% === Setting actual document properties ===

% We use line-spacing 1.15 in the final version and 2.0 in the manuscripts
\ifthenelse{\boolean{BSRSL@final}}
   {\renewcommand{\baselinestretch}{1.15}}{}
\ifthenelse{\boolean{BSRSL@manuscript}}
   {\renewcommand{\baselinestretch}{2.0}}{}

% Indent 1st line of each par by 0.75cm
\setlength{\parindent}{0.75cm}

% Indent each paragraph, also first one
% \RequirePackage{indentfirst}

% Additional spacing between paragraphs: 6pt
% (we add here +2pt ... -1pt elasticity).
\setlength{\parskip}{6pt plus2pt minus1pt}

% Required font sizes are:
% 16pt for article itle
% 14pt for subtitle and leve 1 section titles
% 12pt for all the rest

% tex/latex/base/size12pt.clo has:
%   \normalsize -> \@setfontsize\normalsize\@xiipt{14.5}\@xiipt{14.5},
%                  i.e., 12pt characters with 14.5pt baselineskip
%   \large      -> \@setfontsize\large{\@setfontsize\large\@xivpt{18}}
%   \Large      -> \@setfontsize\Large\@xviipt{22}
% where (in tex/latex/base/latex.ltx)
%   \@xiipt is set to 12
%   \@xivpt is set to 14.4
%   \@xviipt is set to 17.28
% \normalsize is OK for 12pt
% \large too large for sections => define new size \sectionfontsize
% \Large too large for title => define new size \titlefontsize


% === Definition of markup commands ===

\DeclareRobustCommand\sectionfontsize{\@setfontsize\sectionfontsize{14pt}{18}}
\DeclareRobustCommand\titlefontsize{\@setfontsize\titlefontsize{16pt}{20.5}}
\DeclareRobustCommand\subtitlefontsize{\@setfontsize\subtitlefontsize{14pt}{18}}




% The section numbering scheme has one additional level here
% (was 3 in article.cls)
\setcounter{secnumdepth}{4}

% Put point after section numbers
% Avoids having to redefine all the \the...
% (according to https://texfaq.org/FAQ-seccntfmt)
\renewcommand*{\@seccntformat}[1]{%
  \csname the#1\endcsname.\quad
}

% Usage of \@startsection: see https://latexref.xyz/_005c_0040startsection.html
% \@startsection{name}
%   {level}
%   {indent}
%   {beforeskip}  % <= if beforeskip < 0, 1st par not indented, else indented 
%   {afterskip}   % <= if afterskip < 0, run in title, else vspace
%   {style}*[toctitle]{title}


% Section: 1. ... Times New Roman, bold, 14pt
%   is as follows in tex/latex/base/article.cls:
%   \newcommand\section{\@startsection {section}{1}{\z@}%
%                                      {-3.5ex \@plus -1ex \@minus -.2ex}%
%                                      {2.3ex \@plus.2ex}%
%                                      {\normalfont\Large\bfseries}}
\renewcommand\section{\@startsection{section}{1}{\z@}%
                                   {3.5ex \@plus 1ex \@minus .2ex}%
                                   {0.01pt \@plus.2ex}%
                                   {\normalfont\sectionfontsize\bfseries}}

% Subsection: 1.1. ... Times New Roman, bold, 12pt
%   is as follows in tex/latex/base/article.cls:
%   \newcommand\subsection{\@startsection{subsection}{2}{\z@}%
%                                        {-3.25ex\@plus -1ex \@minus -.2ex}%
%                                        {1.5ex \@plus .2ex}%
%                                        {\normalfont\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {3.25ex\@plus 1ex \@minus .2ex}%
                                     {0.01pt \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}

% Subsubsection: 1.1.1. ...  Times New Roman, italics, 12pt
%   is as follows in tex/latex/base/article.cls:
%   \newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
%                                        {-3.25ex\@plus -1ex \@minus -.2ex}%
%                                        {1.5ex \@plus .2ex}%
%                                        {\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {3.25ex\@plus 1ex \@minus .2ex}%
                                     {0.01pt \@plus .2ex}%
                                     {\normalfont\normalsize\itshape}}

% Subsubsubsection: 1.1.1.1. ...  Times New Roman, upright, 12pt
%  no such thing in tex/latex/base/article.cls
%  so, first create counter
\newcounter{subsubsubsection}[subsubsection]
%  \thesubsubsubsection appears to be defined already once
%  the subsubsubsection counter is created
\renewcommand\thesubsubsubsection{\thesubsubsection.\@arabic\c@subsubsubsection}
\newcommand\subsubsubsection{\@startsection{subsubsubsection}{4}{\z@}%
                                     {3.25ex\@plus 1ex \@minus .2ex}%
                                     {0.01pt \@plus .2ex}%
                                     {\normalfont\normalsize\upshape}}
% The two following are required because, if they are omitted, strange effects take place.
\newcommand*\l@subsubsubsection{\@dottedtocline{3}{10.0em}{4.1em}}
\newcommand*{\subsubsubsectionmark}[1]{}
%TBD% Hyperref complaint regarding \subsubsubsection to be solved

% Paragraph (not in Word and ODT styles
% We redefine it here to stay consistent with secnumdepth counting
%   is as follows in tex/latex/base/article.cls:
%   \newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
%                                       {3.25ex \@plus1ex \@minus.2ex}%
%                                       {-1em}%
%                                       {\normalfont\normalsize\bfseries}}
\renewcommand\theparagraph{\thesubsubsubsection.\arabic{paragraph}}
\renewcommand\paragraph{\@startsection{paragraph}{5}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}

% Paragraph (not in Word and ODT styles)
% We redefine it here to stay consistent with secnumdepth counting
%   is as follows in tex/latex/base/article.cls:
%   \newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
%                                          {3.25ex \@plus1ex \@minus .2ex}%
%                                          {-1em}%
%                                         {\normalfont\normalsize\bfseries}}
\renewcommand\subparagraph{\@startsection{subparagraph}{6}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                       {\normalfont\normalsize\bfseries}}

% We need:
% Title: centered, Times New Roman, bold, 16pt
% Subtitle: centred, Times New Roman, bold, 14pt
%   add \ifsubtitle, preset to false, reset to true if the \subtitle command is issued
% Authors
%  - Single author may have more than one affiliation
%  - With single author: no corresponding author mark
%  - If several corresponding authors: change text to "Corresponding authors"
%    / "Correspondants"
% Affiliations and addresses
%  - centred if BSRSL@nb@affils equals 1, no mark.
%  - left aligned if BSRSL@nb@affils > 1


% === Author and affiliations ===

% --- Bookkeeping ---
\newcounter{BSRSL@nb@authors} % must be > 0 at the end
\newcounter{BSRSL@nb@cofirst} % must be 0 or 2 at the end
\newcounter{BSRSL@nb@corraut} % must be 0 or 2 at the end
\newcounter{BSRSL@nb@deceased} 

% Define the keys in the "authopt" family
% Note: the "default" values in brackets are used only when the key is
% included in the call, but without an explicit value assignment.
% Pre-defined key values remain assigned for all keys that are
% not included in a \setkeys call.
\define@key{authopt}{affil}{\def\authopt@affil{#1}}
\define@key{authopt}{corresponding}[true]{\def\authopt@corrauth{#1}}
\define@key{authopt}{cofirst}[true]{\def\authopt@cofirst{#1}}
\define@key{authopt}{deceased}[true]{\def\authopt@deceased{#1}} % must not be combined with "corresponding"


% Author list
% -----------
%
% Temporary lists and storage
% BSRSL@AllAuthorsButLast
% BSRSL@CurrentLastAuthor
% - start with empty
% - first \author[]{}{}:
%   - leave BSRSL@AllAuthorsButLast empty
%   - typeset first author into BSRSL@CurrentLastAuthor
% - second author[]{}{}:
%   - move BSRSL@CurrentLastAuthor to BSRSL@AllAuthorsButLast
%   - typeset second author into BSRSL@CurrentLastAuthor
% - third and following \author[]{}{}
%   - append ", " to BSRSL@AllAuthorsButLast
%     and then append BSRSL@CurrentLastAuthor to BSRSL@AllAuthorsButLast
%   - typeset new author into BSRSL@CurrentLastAuthor
%
% Final typesetting:
% - if BSRSL@nb@authors equals 1 (or BSRSL@AllAuthorsButLast empty),
%   typeset BSRSL@CurrentLastAuthor
% - else typeset BSRSL@AllAuthorsButLast followed by " and " followed by
%   BSRSL@CurrentLastAuthor

\gdef\BSRSL@AllAuthorsButLast{\empty}
\gdef\BSRSL@CurrentLastAuthor{\empty}
\gdef\BSRSL@AddressCorrespondence{\empty}
\newboolean{BSRSL@affilsempty}


% Prepare the \author command, calling upon the internal \@author
\def\author{\@ifnextchar[\@author{\@author[]}}
\def\@author[#1]#2#3{%
%%%%%    \noindent --- @author start\newline%
   % set the keys to the default values to be used for the missing keys
   \setkeys{authopt}{affil=\empty,corresponding=false,cofirst=false,deceased=false}
   % and update to conform to the options set in #1
   \setkeys{authopt}{#1}%
%%%%%   -- key values\newline
%%%%%   First name = #2\newline
%%%%%   Last name = #3\newline
%%%%%   affiliation = \authopt@affil\newline
%%%%%   cofirst = \authopt@cofirst\newline
%%%%%   corrauth = \authopt@corrauth\newline
%%%%%   deceased = \authopt@deceased\newline
   \@addauthortolist{#2}{#3}{\authopt@affil}{\authopt@cofirst}{\authopt@corrauth}{\authopt@deceased}
%%%%%   \noindent All but last = +\BSRSL@AllAuthorsButLast+\newline
%%%%%   \noindent Current last = +\BSRSL@CurrentLastAuthor+\newline
%%%%%   --- @author end\par%
}

\def\@addauthortolist#1#2#3#4#5#6{%
   % args:
   % #1 First name
   % #2 Last name
   % #3 \authopt@affil
   % #4 \authopt@cofirst
   % #5 \authopt@corrauth
   % #6 \authopt@deceased
   %
   %%%%%   --- @addauthortolist start\newline
   \gdef\@note@corraut{}
   \gdef\@note@cofirst{}
   \addtocounter{BSRSL@nb@authors}{1}
   \ifthenelse{\equal{#4}{true}}{\addtocounter{BSRSL@nb@cofirst}{1}}{}
   \ifthenelse{\equal{#5}{true}}{\addtocounter{BSRSL@nb@corraut}{1}}{}
   \ifthenelse{\equal{#6}{true}}{\addtocounter{BSRSL@nb@deceased}{1}}{}
   %
   \ifthenelse{\value{BSRSL@nb@authors}=1}%
   % Process the first given author
   {% - first check if the author's affiliation is empty: 
      \ifthenelse{\equal{#3}{\empty}}%
      {% start of "affil for first author is empty"
       % if so, all the others will have to be empty as well
         \setboolean{BSRSL@affilsempty}{true}
         % if the first author is co-first, we have to take extra
         % care about commas.
         \ifthenelse{\equal{#4}{true}}% is the first author co-first?
         {% start of "first author is co-first"
            \def\@note@cofirst{*}%
            \ifthenelse{\equal{#5}{true}}{\def\corrauthnote{,\noexpand\ast}}{}
         }% end of "first author is co-first"
         {% start of "first author is not co-first"
            \ifthenelse{\equal{#5}{true}}{\def\corrauthnote{\noexpand\ast}}{}
         }% end of "first author is not co-first"
      }% end of "affil for first author is empty"
      {% start of "affil for first author is not empty"
       % if so, all the later ones will have to be non-empty as well
         \setboolean{BSRSL@affilsempty}{false}
         \ifthenelse{\equal{#4}{true}}{\def\@note@cofirst{,\noexpand\S}}{}
         \ifthenelse{\equal{#5}{true}}{\def\@note@corraut{,\noexpand\ast}}{}
      }% end of "affil for first author is not empty"
      \edef\@temp@CLA@name{\expand@once{#1}\ \noexpand\textsc{\expand@once{#2}}}
      \edef\@temp@CLA@note{#3\@note@cofirst\@note@corraut}
      \edef\@temp@CLA{\noexpand\mbox{\expand@once{\@temp@CLA@name}$^{\expand@once{\@temp@CLA@note}}$}}
%%%%%       \typeout{First author CLA: \@temp@CLA}
%%%%%       \show\@temp@CLA
      \let\BSRSL@CurrentLastAuthor=\@temp@CLA
   }% end of "process the first author" 
   {% start of "process the second and following author(s)"
      \ifthenelse{\value{BSRSL@nb@authors}=2}
      {  \ifthenelse{\equal{#4}{true}}
         {  \ifthenelse{\value{BSRSL@nb@cofirst}=2}
            {}{\ClassError{bullsrsl[@addauthortolist]}
                          {The second author cannot be co-first}
                          {The first author was not flagged co-first,\MessageBreak
                           so the second one cannot be co-first either.}%
            }%
         }
         {  \ifthenelse{\value{BSRSL@nb@cofirst}=1}
            {  \ClassError{bullsrsl[@addauthortolist]}
                          {The second author must be co-first}
                          {The first author was flagged co-first,\MessageBreak
                           so the second one must be co-first as well.}%
            }{}%
         }%
      }
      {  \ifthenelse{\equal{#4}{true}}% is the third or later author flagged co-first?
         {  \ClassError{bullsrsl[@addauthortolist]}
                       {This author cannot be co-first}
                       {Only the two first authors may be considered co-first.}
         }{}%
      }
      \ifthenelse{\equal{#3}{\empty}}%
      {  \ifthenelse{\boolean{BSRSL@affilsempty}}
         {% should be empty, is empty - OK to proceed
            \ifthenelse{\equal{#4}{true}}% is the (second) author co-first?
            {  \def\@note@cofirst{\noexpand\S}%
               \ifthenelse{\equal{#5}{true}}{\def\@note@corraut{,\noexpand\ast}}{}
            }% end of "author is co-first"
            {  \def\@note@cofirst{}%
               \ifthenelse{\equal{#5}{true}}{\def\@note@corraut{\noexpand\ast}}{}
            }% end of "author is not co-first"
         }
         {% should not be empty, but is empty - error:
            \ClassError{bullsrsl[@addauthortolist]}
                       {Affiliation is empty}
                       {Affiliations for all authors must be explicit\MessageBreak
                        since the affiliation for the first author was explicit.}%
         }%
      }
      {  \ifthenelse{\boolean{BSRSL@affilsempty}}
         {% should be empty, but is not empty - error:
            \ClassError{bullsrsl[@addauthortolist]}
                       {Affiliation is not empty}
                       {Affiliations for all authors must be left empty\MessageBreak
                        since the affiliation for the first author was empty.}%
         }
         {% should not be empty, is not empty - OK to proceed
            \ifthenelse{\equal{#4}{true}}{\def\@note@cofirst{,\noexpand\S}}{}
            \ifthenelse{\equal{#5}{true}}{\def\@note@corraut{,\noexpand\ast}}{}
         }%
      }
      \ifthenelse{\value{BSRSL@nb@authors}=2}
      {% We are processing the second author in the list
         \let\BSRSL@AllAuthorsButLast=\BSRSL@CurrentLastAuthor%
%%%%%        \typeout{Second author AABL: \BSRSL@AllAuthorsButLast}
%%%%%        \show\BSRSL@AllAuthorsButLast
      }
      {% We are processing the third or later author in the list
         \edef\@temp@AABL{\expand@once{\BSRSL@AllAuthorsButLast}, \expand@once{\BSRSL@CurrentLastAuthor}}%
         \let\BSRSL@AllAuthorsButLast=\@temp@AABL%
%%%%%        \typeout{Third and later author AABL: \BSRSL@AllAuthorsButLast}
%%%%%        \show\BSRSL@AllAuthorsButLast
      }
      \edef\@temp@CLA@name{\expand@once{#1}\ \noexpand\textsc{\expand@once{#2}}}
      \edef\@temp@CLA@note{#3\@note@cofirst\@note@corraut}
      \edef\@temp@CLA{\noexpand\mbox{\expand@once{\@temp@CLA@name}$^{\expand@once{\@temp@CLA@note}}$}}
%%%%%     \typeout{Second and later author CLA: \@temp@CLA}
%%%%%     \show\@temp@CLA
      \let\BSRSL@CurrentLastAuthor=\@temp@CLA
   }% end of "process the second and following author(s)"
%%%%%   --- @addauthortolist end\newline
}

% The following only works with a single corresponding author.
%\newcommand*\correspondance[1]{%
%   \def\BSRSL@CorrEmail{\href{mailto:#1}{#1}}%
%}

\newcommand*\correspondance[1]{%
   \def\BSRSL@CorrEmail{#1}%
}


\gdef\BSRSL@AffiliationList{\empty}
\newcounter{BSRSL@nb@affils}
\newcounter{@temp@nb@affils}


\def\affiliation{\@ifnextchar[\@affiliation{\@affiliation[]}}
\def\@affiliation[#1]#2{%
   % First of all deactivate the author management commands
   % so that no more authors can be added
   \global\let\author\relax
   \global\let\@author\relax
   \global\let\@addauthortolist\relax
   %
   % Then check if the affiliation reference is empty. This is only acceptable
   % if all authors' affiliation references are empty.
   \ifthenelse{\equal{#1}{\empty}}
   {% The reference for this affiliation is empty
      \ifthenelse{\boolean{BSRSL@affilsempty}}
      {% All must be empty anyway - OK.
       % However, in this case, there can only be one affiliation
         \ifthenelse{\value{BSRSL@nb@affils}=0}
         {% This is the first affiliation that we process - OK, proceed
            \setcounter{BSRSL@nb@affils}{1}%
         }
         {% There has been another one processed before:
            \ClassError{bullsrsl[@affiliation]}
                       {Spurious extra affiliation}
                       {No author had an "affil" key set. In this case, you may\MessageBreak
                        specify only one affiliation that will be common for all.\MessageBreak
                        Please remove all affiliations but one, or include\MessageBreak
                        appropriate "affil" keys in the author definitions.}%
         }%
      }
      {% No affiliation reference can be empty, but this one is:
         \ClassError{bullsrsl[@affiliation]}
                    {Missing affiliation reference}
                    {This affiliation does not include any reference\MessageBreak
                     although all authors require affiliations.\MessageBreak
                     Please provide a reference.}%
      }%
   }
   {% The reference for this affiliation is not empty
      \ifthenelse{\boolean{BSRSL@affilsempty}}
      {% However, all must be empty
         \ClassError{bullsrsl[@affiliation]}
                    {Unallowed affiliation reference}
                    {No author requires an explicit affiliation and\MessageBreak
                     therefore, the optional reference must be left empty.}%
      }
      {% OK - affiliation references are required.
       % Check if it is numeric (implicit check, as we get an error below
       % if it is not) and if its number follows the previous one
         \setcounter{@temp@nb@affils}{\value{BSRSL@nb@affils}}
         \addtocounter{@temp@nb@affils}{1}
         \ifthenelse{\equal{#1}{\arabic{@temp@nb@affils}}}
         {% Reference acceptable: increment counter (accept it)
            \addtocounter{BSRSL@nb@affils}{1}%
         }
         {  \ClassError{bullsrsl[@affiliation]}
                       {Non-numeric or non continuous affiliation reference}
                       {Affiliation references must be numerical.}%
         }%
      }%
   }%
   % If the current affiliation is not the first one, check if the authors'
   % affiliation references must not be empty. In this case, only one
   % affiliation is allowed.
   \ifthenelse{\value{BSRSL@nb@affils}>1}
   {% If this is not the first affiliation, check if affiliation references
    % must not be empty.
      \ifthenelse{\boolean{BSRSL@affilsempty}}
      {% All authors' affiliations must be empty
         \ClassError{bullsrsl[@affiliation]}
                    {Unallowed extra affiliation}
                    {None of the authors requires an explicit affiliation.\MessageBreak
                     In this case, only one affiliation is allowed.}%
      }
      {% OK, all authors require affiliations.
      }%
   }
   {}%
   % If all affiliations are empty, the affiliation reference note remains
   % empty as well; if not, it is set to the reference number
   \ifthenelse{\boolean{BSRSL@affilsempty}}
   {% If all affiliations are empty, there is actually only one, and we can
    % generate \BSRSL@AffiliationList right away
      \edef\@temp@affil@address{\expand@once{#2}\noexpand}
      \let\BSRSL@AffiliationList=\@temp@affil@address
   }
   {% Affiliations are not empty: create the note
    % and append the address for this affiliation
      \edef\@temp@affil@note{$^{#1}$}
      \edef\@temp@affil@address{\expand@once{#2}}
      \edef\@temp@affil{\expand@once{\@temp@affil@note}~\expand@once{\@temp@affil@address}}%
      \ifthenelse{\value{BSRSL@nb@affils}=1}
      {% We are processing the first affiliation:
       % start the list with the current affiliation
         \let\BSRSL@AffiliationList=\@temp@affil%
      }
      {% This is not the first one: append a \newline and then
       % the current affiliation to the existing non empty list.
         \edef\@temp@AL{\expand@once{\BSRSL@AffiliationList}\noexpand\newline\ \expand@once{\@temp@affil}}%
         \let\BSRSL@AffiliationList=\@temp@AL%
      }%
   }%
}



% === Article Title ===


% --- Main Title ---

% The following has been adapted and simplified from the Standard LaTeX article
% class (file texmf-dist/tex/latex-dev/base/article.cls, version v1.4n from
% 2021/10/04), lines 203--253
\renewcommand\maketitle{\par
   \@maketitle@finalpreps
   \begingroup
      \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
      \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
      \long\def\@makefntext##1{\parindent 1em\noindent
         \hb@xt@1.8em{%
            \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
      \thispagestyle{plain}\@thanks
   \endgroup
   \setcounter{footnote}{0}%
   \global\let\thanks\relax
   \global\let\maketitle\relax
   \global\let\@maketitle\relax
   \global\let\@thanks\@empty
   \global\let\@author\@empty
   \global\let\@date\@empty
   \global\let\@title\@empty
   \global\let\title\relax
   \global\let\author\relax
   \global\let\date\relax
   \global\let\and\relax
}


\def\@maketitle{%
   \newpage
   \null
   \vskip 2em%
   \begin{center}%
   \let \footnote \thanks
   {\titlefontsize\bfseries \@title \par}%
   %{\subtitlefontsize\bfseries \@subtitle \par}%
   %GM: commented out to remove unwanted extra space%\vskip 1.5em%
   {\normalsize
      \lineskip .5em%
      %GM: commented out \begin{tabular}[t]{c}%
      %GM: commented out   \@author
      %GM: commented out \end{tabular}
      \ifthenelse{\value{BSRSL@nb@authors}>1}
      {  \BSRSL@AllAuthorsButLast\ \mbox{\and@et\ \BSRSL@CurrentLastAuthor}}
      {  \BSRSL@CurrentLastAuthor}\par%
   }%
   \vskip 1em%
   %GM: commented to remove unwanted \date%{\large \@date}%
   \end{center}%
   \par
   \ifthenelse{\value{BSRSL@nb@affils}=1}
   {% all centered
      \begin{center}
         \BSRSL@AffiliationList\\
         \BSRSL@CorrespIntro~\BSRSL@CorrEmail
      \end{center}
   }
   {% left aligned
      \noindent%
      \BSRSL@AffiliationList\newline
      \ifthenelse{\value{BSRSL@nb@cofirst}>0}
      {  \ifthenelse{\boolean{BSRSL@french}}
            {% in French
               $^{\S}$~Co-premiers auteurs}
            {% in English
               $^{\S}$~Co-first authors}%
         \newline%
      }{}%
      \BSRSL@CorrespIntro~\BSRSL@CorrEmail%
   }%
%%%%% Debugging info:
%%%%%      Number of authors: \theBSRSL@nb@authors\newline
%%%%%      Number of co-first: \theBSRSL@nb@cofirst\newline
%%%%%      Number of corresp. authors: \theBSRSL@nb@corraut\newline
%%%%%      Number of deceased: \theBSRSL@nb@deceased\newline
%%%%%      Number of addresses: \theBSRSL@nb@affils\newline
%%%%%   \vskip 1.5em%
}


\newcommand\@maketitle@finalpreps{%
   % Are there any authors declared?
   \ifthenelse{\value{BSRSL@nb@authors}=0}
   {% No authors have been declared
      \ClassError{bullsrsl[@maketitle@finalpreps]}
                 {Missing author(s)}
                 {No author has been declared.}}
   {% Author(s) found - OK, proceed.
   }%
   % Are there any affiliations declared?
   \ifthenelse{\value{BSRSL@nb@affils}=0}
   {% No affiliations have been declared
      \ClassError{bullsrsl[@maketitle@finalpreps]}
                 {Missing affiliations(s)}
                 {No affiliation has been declared.}}
   {% Affiliation(s) found - OK, proceed.
   }%
   %
   % If there is only one author:
   % - it should not have any "affil" key, except if
   %   the author has several affiliations
   % - it cannot be co-first author (no "cofirst" key)
   % - it should not have any "corresponding" key
   % - it's correspondance information line starts with
   %   "Correspondance to:"
   %
   % If there are two authors
   % - the second one can only be co-first if the first one is also co-first
   %   (already checked)
   % - at least one of them must have the "corresponding" key
   % - their correspondance information starts with
   %   "Corresponding author:" if there is only one corresponding author,
   %   and "Corresponding authors:" if there are several of them.
   %   ("Auteur correspondant:", resp. "Auteurs correspondants:" in french).
   \ifthenelse{\value{BSRSL@nb@authors}=1}
   {% There is only one author...
    % - check if there is only one affiliation
      \ifthenelse{\value{BSRSL@nb@affils}=1}
      {% there is only one affiliation:
       % OK if the affiliation note is empty
         \ifthenelse{\boolean{BSRSL@affilsempty}}
         {% Affiliations are empty - OK, proceed
         }
         {% Affiliations are not empty
            \ClassError{bullsrsl[@maketitle@finalpreps]}
                       {Unwanted affiliation reference}
                       {For a single author manuscript, no "affil" key must be\MessageBreak
                        used in the author declaration and no affiliation reference\MessageBreak
                        in the affiliation declaration.\MessageBreak
                        Please remove the "affil" key from the author declaration\MessageBreak
                        and the affiliation reference from the affiliation declaration}%
         }
      }%
      {% more than one - OK proceed
      }
      % - check for co-first characterisation
      \ifthenelse{\value{BSRSL@nb@cofirst}=0}
      {% No co-first authors - OK proceed
      }
      {% Spurious co-first characterisation
         \ClassError{bullsrsl[@maketitle@finalpreps]}
                    {Unallowed co-first author}
                    {A single author cannot be co-first author.\MessageBreak
                     Please remove the "cofirst" key from the author declaration.}%
      }%
      % - check for "correspondance" key
      \ifthenelse{\value{BSRSL@nb@corraut}=0}
      {% No "corresponding" key included - OK, proceed
       % Define correspondance introduction text
         \ifthenelse{\boolean{BSRSL@french}}
         {% in French
            \def\BSRSL@CorrespIntro{Correspondance {\`a}:\ }}
         {% in English
            \def\BSRSL@CorrespIntro{Correspondance to:\ }}%
      }
      {% A spurious "corresponding" key is included:
         \ClassError{bullsrsl[@maketitle@finalpreps]}
                    {Unnecessary "corresponding" key}
                    {For single author manuscripts, the "corresponding" key\MessageBreak
                     should be left out. Please remove it.}%
      }%
   }
   {% There is more than one author...
    % - check if one of them at least has a "corresponding" key
      \ifthenelse{\value{BSRSL@nb@corraut}=0}
      {% No author has got a "corresponding" key: error
         \ClassError{bullsrsl[@maketitle@finalpreps]}
                    {Missing corresponding author}
                    {No author has been selected as a corresponding author.\MessageBreak
                     This is only allowed for single author manuscripts.\MessageBreak
                     Please select one or more corresponding authors.}%
      }
      {% One or more authors have got a "corresponding" key.
       % Adapt the correspondance introduction text
       % depending on the number of corresponding authors
         \ifthenelse{\value{BSRSL@nb@corraut}=1}
         {  \ifthenelse{\boolean{BSRSL@french}}
            {% in French
               \def\BSRSL@CorrespIntro{$^{\ast}$~Auteur correspondant:\ }}
            {% in English
               \def\BSRSL@CorrespIntro{$^{\ast}$~Corresponding author:\ }}%
         }
         {  \ifthenelse{\boolean{BSRSL@french}}
            {% in French
               \def\BSRSL@CorrespIntro{$^{\ast}$~Auteurs correspondants:\ }}
            {% in English
               \def\BSRSL@CorrespIntro{$^{\ast}$~Corresponding authors:\ }}%
         }%
      }%
   }%
   %
   \ifthenelse{\value{BSRSL@nb@affils}=1}
   {% If there is only one affiliation, the affiliation
    % reference(s) for all authors must not be specified.
      \ifthenelse{\boolean{BSRSL@affilsempty}}
      {% all are empty - OK, proceed.
      }
      {% they are not empty, but better had to be.
         \ClassError{bullsrsl[@maketitle@finalpreps]}
                    {Author affiliation references not required}
                    {There is only one affiliation and therefore, none of\MessageBreak
                     the author specifications should include an "affil" key.}%
      }%
   }
   {% There is more than one affiliation - OK, proceed
    % Notice that even a single author may have mutiple affiliations.
    % and therefore, the warning is only triggered if there is a 
   }%
}


% --- Subtitle (optional) ---
% (centered, Times New Roman, bold, 14pt)
% TBD


% === Abstract / Resume ===

% If the main language of the paper is French, we require an alternative
% abstract in English (entitled "Abstract"); if the main language is English, an
% second abstract in French may be added, entitled "Résumé", but is not mandatory.
% If the second abstract is given though, the keywords must also be translated.
\ifthenelse{\boolean{BSRSL@french}}
{  \newcommand\altabstractname{Abstract}}
{  \newcommand\altabstractname{R\'esum\'e}}%

\ifthenelse{\boolean{BSRSL@french}}
{  \renewcommand\abstractname{R\'esum\'e}}
{  \renewcommand\abstractname{Abstract}}%


%\renewenvironment{abstract}{%
%    \begin{raggedright}
%    {\normalfont\large\bfseries\abstractname\vspace{-\parskip}\vspace{\z@}}%
%    \end{raggedright}\par\noindent\ignorespaces}{}
\renewenvironment{abstract}{%
   \section*{\abstractname}
   \noindent\ignorespaces}{\par}

%\newenvironment{altabstract}{%
%    \begin{raggedright}
%    {\normalfont\large\bfseries\altabstractname\vspace{-\parskip}\vspace{\z@}}%
%    \end{raggedright}\par\noindent\ignorespaces}{}
\newenvironment{altabstract}{%
   \section*{\altabstractname}
   \noindent\ignorespaces}{\par}

\ifthenelse{\boolean{BSRSL@french}}
{  \def\BSRSL@KeywordIntro{Mots-cl\'es~:\ }%
   \def\BSRSL@MSCIntro{MSC~:\ }%
   \def\BSRSL@AltKeywordIntro{Keywords:\ }}
{  \def\BSRSL@KeywordIntro{Keywords:\ }%
   \def\BSRSL@MSCIntro{MSC:\ }%
   \def\BSRSL@AltKeywordIntro{Mots-cl\'es~:\ }}%

\newboolean{BSRSL@keywordsgiven}

\newcommand*{\keywords}[1]{%
   \noindent\ignorespaces\textbf{\BSRSL@KeywordIntro} {#1}\par%
}

\newcommand*{\msccodes}[1]{%
   \noindent\ignorespaces\textbf{\BSRSL@MSCIntro} {#1}\par%
}

\newcommand*{\altkeywords}[1]{%
   \noindent\ignorespaces\textbf{\BSRSL@AltKeywordIntro} {#1}\par%
}


% === Acknowledgments (optional) ===

\ifthenelse{\boolean{BSRSL@french}}
{  \newcommand\acknowlname{Remerciements}}
{  \newcommand\acknowlname{Acknowledgments}}

\newenvironment{acknowledgments}{%
   \section*{\acknowlname}
   \noindent\ignorespaces}{}


% === Further information (mandatory) ===

\newboolean{BSRSL@furthinfopen}

\newcounter{BSRSL@nb@furthinf}
\newcounter{BSRSL@nb@orcids}
\newcounter{BSRSL@nb@authctrb}
\newcounter{BSRSL@nb@conflint}


\ifthenelse{\boolean{BSRSL@french}}
{  \newcommand\furinfname{Informations suppl\'ementaires}
   \ifthenelse{\value{BSRSL@nb@authors}=1}
   {  \newcommand\orcidsname{Identifiant ORCID de l'auteur}}
   {  \newcommand\orcidsname{Identifiants ORCID des auteurs}}
   \newcommand\authctrbname{Contributions des auteurs}
   \newcommand\conflintname{Conflits d'int\'er\^et}}
{  \newcommand\furinfname{Further Information}
   \ifthenelse{\value{BSRSL@nb@authors}=1}
   {  \newcommand\orcidsname{ORCID identifier of the author}}
   {  \newcommand\orcidsname{ORCID identifiers of the authors}}
   \newcommand\authctrbname{Author contributions}
   \newcommand\conflintname{Conflicts of interest}}


\newenvironment{furtherinformation}
{%
   \section*{\furinfname}
   \setboolean{BSRSL@furthinfopen}{true}%
   \noindent\ignorespaces%
}
{%
   \setboolean{BSRSL@furthinfopen}{false}%
   \ignorespacesafterend%
}


% --- ORCID identifiers of authors (optional) ---

\newenvironment{orcids}
{%
   \ifthenelse{\boolean{BSRSL@furthinfopen}}
   {%
      \subsection*{\orcidsname}%
      \noindent\ignorespaces%
   }%
   {% Author contribution given with single author
      \ClassError{bullsrsl[orcids]}
                 {orcids environment outside furtherinformation}
                 {The "orcids" environment must be used inside\MessageBreak
                  the "furtherinformation" environment.\MessageBreak
                  Please use \begin{furtherinformation} first.}%
   }%
}
{%
   \ignorespacesafterend%
}

\newcounter*
\newcommand*{\orcid}[3]{%
   \ifthenelse{\value{BSRSL@nb@orcids}>0}
   {%  I not first one, put out a \newline first, else nothing
       \newline}{}%
   \noindent\ignorespaces\href{https://orcid.org/#1}{#1} ({#2}\nobreakspace\textsc{#3})%
   \addtocounter{BSRSL@nb@orcids}{1}
}

% --- Author contributions (MANDATORY) ---

\newenvironment{authorcontributions}
{%
   \ifthenelse{\boolean{BSRSL@furthinfopen}}
   {% OK, Further Information has been begun.
   }
   {% Author contributions information  outside \begin/\end{furtherinformation}
      \ClassError{bullsrsl[authorcontributions]}
                 {authorcontributions environment outside furtherinformation}
                 {The "authorcontributions" environment must be used inside\MessageBreak
                  the "furtherinformation" environment.\MessageBreak
                  Please issue \begin{furtherinformation} first.}%
   }%
   \ifthenelse{\value{BSRSL@nb@authors}=1}
   {% Author contribution given with single author
      \ClassError{bullsrsl[authorcontributions]}
                 {Unwanted authorcontributions}
                 {For a single author manuscript, no author contribution information\MessageBreak
                  must be given. Please remove the "authorcontributions" environment.}%
   }
   {%
         \subsection*{\authctrbname}
         \noindent\ignorespaces%
   }%
}
{%
   \setcounter{BSRSL@nb@authctrb}{1}%
   \ignorespacesafterend%
}


% --- Conflicts of interest (MANDATORY) ---

\newenvironment{conflictsofinterest}
{%
   \ifthenelse{\boolean{BSRSL@furthinfopen}}
   {%
      \subsection*{\conflintname}
      \noindent\ignorespaces%
   }
   {% Conflicts of interest outside \begin/\end{furtherinformation}
      \ClassError{bullsrsl[conflictsofinterest]}
                 {"conflictsofinterest" environment outside "furtherinformation"}
                 {The "conflictsofinterest" environment must be used inside\MessageBreak
                  the "furtherinformation" environment.\MessageBreak
                  Please issue \begin{furtherinformation} first.}%
   }%
}
{%
   \setcounter{BSRSL@nb@conflint}{1}%
   \ignorespacesafterend%
}



%%% TEST STUFF - can probably be discarded
%\newenvironment{altabstract}{%
%    \begin{raggedright}
%    {\normalfont\large\bfseries\altabstractname\vspace{-\parskip}\vspace{\z@}}%
%    \end{raggedright}\par\noindent\ignorespaces}{}


% === Supporting information (optional) ===
% - \section* level

% === References ===
% (should be \subsection* is \section in natbib)
% --> reset within \AtBeginDocument (also change \refname as a function of language)
\AtBeginDocument{%
   \ifthenelse{\boolean{BSRSL@french}}
   {  \renewcommand\refname{R\'ef\'erences}}
   {  \renewcommand\refname{References}}%
}


% === Figures and Tables ===

\ifthenelse{\boolean{BSRSL@french}}
{  \renewcommand\tablename{Tableau}}{}

% Figure caption 2cm away from left and from righthand margins,
% centered if one-liner, justified else
% TBD
% See https://tex.stackexchange.com/questions/422359/how-to-set-width-of-caption-to-width-of-figure
% for ideas about solving the caption width required for tables.
% The following has been adapted and simplified from the Standard LaTeX article
% class (file texmf-dist/tex/latex-dev/base/article.cls, version v1.4n from
% 2021/10/04), lines 480-489:
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{{\bf #1:} #2}%
  \ifdim \wd\@tempboxa >\hsize
    {\bf #1:} #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}




% Switch on line numbers if we are in 'manuscript' mode
\ifthenelse{\boolean{BSRSL@manuscript}}
{  \linenumbers}
{}

\newcommand*{\check@authctrb}{%
   \ifthenelse{\value{BSRSL@nb@authors}=1}
   {% No author's contribution required - proceed
      No author's contribution required\newline
   }
   {% More than one author
      \ifthenelse{\value{BSRSL@authctrb}=1}
      {% Author contributions have been given. OK, green light.
         Author contributions have been given\newline
      }
      {% Authors contributions missing
         Authors contributions missing\newline
%%%         \ClassError{bullsrsl[authorcontributions]}
%%%                    {Missing author contributions}
%%%                    {The author contributions have not been specified.\MessageBreak
%%%                     Please given them in an "authorcontributions" environment,\MessageBreak
%%%                     to be included in the "furtherinformation" part.}%
      }%
   }%
}

\newcommand*{\check@conflint}{%
   \ifthenelse{\value{BSRSL@nb@conflint}=1}
   {% Conflicts of interest have been declared. OK, green light.
   }
   {% Conflicts of interest declaration missing
      \ClassError{bullsrsl[conflictsofinterest]}
                 {Missing declaration of conflicts of interest}
                 {The declaration relative to conflicts of interest is missing.\MessageBreak
                  Please declare them in a "conflictsofinterest" environment,\MessageBreak
                  to be included in the "furtherinformation" part.}%
   }%
}

\AtEndDocument{\relax}


