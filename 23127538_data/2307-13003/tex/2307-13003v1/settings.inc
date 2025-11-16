\makeatletter
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\makeatother
\makeatletter
\newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@}
\makeatother

% THEOREM Environments (Examples)----------------------------------------- 
%
\newtheorem{definition}{Definition}[section]
\newtheorem{theorem}[definition]{Theorem}
\newtheorem{lemma}[definition]{Lemma}
\newtheorem{corollary}[definition]{Corollary}
\newtheorem{example}[definition]{Example}
\newtheorem{principle}[definition]{Principle}
\newtheorem{remark}[definition]{Remark}
%
\newcommand{\QED}{\nobreak \ifvmode \relax \else
      \ifdim\lastskip<1.5em \hskip-\lastskip
      \hskip1.5em plus0em minus0.5em \fi \nobreak
      \vrule height0.75em width0.5em depth0.25em\fi}
\newcommand{\bm}[1]{{\mbox{\boldmath $#1$}}}

%
\def\slash#1{\not\!#1}
\def\sr#1{\hspace{-.1em}#1\hspace{-.2em}}
\def\sro#1{\hspace{-.1em}#1\hspace{-.1em}}
\def\LLambda{\begin{large}\Lambda\end{large}}
\def\tgamma{\tilde{\gamma}}
\def\ttheta{\tilde{\theta}}
\def\ET{{E,\vartheta}}
\def\tvq{\tilde{\vartheta}}
%
\def\ItN{{\bm{I}_{\SU^{2}}}}
%
\def\Tr{{\mathrm{Tr}\hspace{-.05em}}}
\def\tn{{2\hspace{-.15em}N}}
%
\def\stars{{\star\star}}

%\def\Rs{{R_{\text{s}}}}
\def\rs{{r_{\text{S}}}}
\def\Rs{{R_\Earth^{\text{S}}}}

\def\diamonds{{\diamond\diamond}}

\def\QED{{\hspace{-.1em}\text{Q}\hspace{-.1em}\text{E}\hspace{-.1em}\text{D}}}

\def\CC{\mathbb{C}}

\def\kE{{\kappa^{~}_\text{E}}}

\def\Nf{{N_{\hspace{-.2em}f}}}
\def\Nnf{{N_{\hspace{-.1em}n\hspace{-.1em}f}}}
\def\Ncf{{N_{\hspace{-.1em}c\hspace{-.1em}f}}}

\def\GR{{G\hspace{-.1em}R}}
\def\MT{{M\hspace{-.1em}T}}
\def\QEGD{{Q\hspace{-.07em}G\hspace{-.12em}E\hspace{-.1em}D}}
\def\QGED{{Q\hspace{-.07em}G\hspace{-.12em}E\hspace{-.1em}D}}
\def\Wedge{{\bm\wedge}}

\def\hlf{{1\hspace{-.11em}/\hspace{-.09em}2}}

\def\bcdot{{\circ}}
\def\bcdots{{\circ\circ}}
\def\bcdott{{\circ\circ\circ\circ}}

\def\lp{l_{\textrm{p}}}
\def\lpt{l_{\textrm{p}}^{\hspace{.3em}2}}
\def\lpf{l_{\textrm{p}}^{\hspace{.3em}4}}

\def\bullets{{\bullet\bullet}}

\def\LGR{{\Lambda^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\LGRt{{\Lambda^{2}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\Gso{{\hat{G}_{\hspace{-.2em}SO}}}
\def\GSO{{G_{\hspace{-.1em}S\hspace{-.1em}O}}}
\def\delBRST{\delta_{\textrm{\hspace{-.1em}B}}}
\def\cGR{{c_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\aGR{{\alpha^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\aGRz{{\alpha^{0}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\bGR{{\beta^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\LLLGF{{\LLL_{G\hspace{-.1em}F}}}
\def\LLLFP{{\LLL_{F\hspace{-.1em}P}}}
\def\LLLQG{{\LLL_{Q\hspace{-.1em}G}}}

%\def\bare{{\hspace{-.1em}0\hspace{-.1em})}}
\def\bare{{\hspace{-.1em}\infty\hspace{-.1em})}}
%\def\bare{\bm{0}}

\def\cSU{{c^{~}_\SU}}
\def\cSUz{{c^\bare_\SU}}

\def\gfix{{g\hspace{-.1em}f\hspace{-.1em}i\hspace{-.1em}x}}

%%%%%%%%%%%%%%%%%%%%%
\def\RII{{I\hspace{-.1em}I}}
\def\DDD{\mathfrak{D}}
\def\M{{\cal M}}
\def\MM{\mathscr{M}}
\def\MMM{\mathfrak{M}}

\def\Tm{{T\hspace{-.2em}M}}
\def\Tsm{{T^*\hspace{-.2em}M}}
\def\TM{{T\hspace{-.2em}\M}}
\def\TsM{{T^*\hspace{-.2em}\M}}
\def\TMM{{T\hspace{-.2em}\MM}}
\def\TsMM{{T^*\hspace{-.2em}\MM}}

\def\Mp{{\M_{\hspace{-.1em}p}}}
\def\TMp{{\TM_{\hspace{-.1em}p}}}
\def\TsMp{{\TsM_{\hspace{-.1em}p}}}
\def\Mg{{\M_{\hspace{-.1em}\theta}}}
\def\TMg{{\TM_{\hspace{-.1em}\theta}}}
\def\TsMg{{\TsM_{\hspace{-.1em}\theta}}}

\def\MMg{{\MM_{\hspace{-.1em}\theta}}}
\def\TMMg{{\TMM_{\hspace{-.1em}\theta}}}
\def\TsMMg{{\TsMM_{\hspace{-.1em}\theta}}}

%%%%%%%%%%%%%%%%%%%%

\def\Gso{{{G}_{\hspace{-.2em}S\hspace{-.1em}O}}}
\def\Gsu{{{G}_{\hspace{-.2em}S\hspace{-.1em}U}}}

\def\TsE{{T^*\hspace{-.2em}E}}

\def\GammaSp{{\Gamma_{\hspace{-.1em}\Sp}}}


\def\AA{{\textbf{A}}}
\def\aaa{{\mathfrak a}}
\def\AAA{{\mathfrak A}}
\def\Aa{{\mathscr{A}}}
\def\aa{{\bm{a}}}

\def\bbb{\mathfrak{b}}
\def\BBB{\mathfrak{B}}

\def\OO{{\mathcal O}}
\def\ooo{\mathfrak{o}}
\def\OOO{\mathfrak{O}}
\def\Fym{\mathfrak{ym}}


\def\r{{\bm{r}}}
\def\tbr{\tilde{\r}}

\def\C{\mathbb{C}}
\def\Z{\mathbb{Z}}
\def\K{\mathbb{K}}

\def\QQQ{\mathfrak{Q}}
\def\qqq{\mathfrak{q}}

\def\uuu{\mathfrak{u}}

\def\yyy{\mathfrak{y}}

\def\XXX{\mathfrak{X}}
\def\X{\mathscr{X}}
\def\xxx{\mathfrak{x}}

\def\ZZZ{\mathfrak{Z}}
\def\zzz{\mathfrak{z}}

\def\rp{\hspace{-.1em}\textrm{q}}
\def\p{\bm{p}}
\def\ppp{{\mathfrak p}}
\def\tppp{\tilde{{\mathfrak p}}}
\def\PPP{{\mathfrak P}}

\def\T{\mathscr{T}}
\def\TT{{\mathcal T}}
\def\TTT{{\mathfrak T}}
\def\ttt{{\mathfrak t}}

\def\d{{\cal D}}
\def\D{{\slash{\cal D}}}
\def\Dn{{\nabla}}
\def\FFF{\mathfrak{D}}
\def\td{d_\www}
\def\ddd{\mathfrak{d}}
\def\ds{{\slash{d}}}

\def\v{{\bm{v}}}
\def\V{{\bm{V}}}
\def\vvv{\mathfrak{v}}
\def\VVV{\mathfrak{V}}


\def\u{{\cal{U}}}
\def\UUU{\mathfrak{U}}

\def\FF{{\mathcal F}}
\def\F{{\bm{F}}}
\def\ff{{\cal F}}
\def\fff{\mathfrak{f}}
\def\FFF{\mathfrak{F}}
\def\f{\mathscr{F}}


\def\g{\mathscr{G}}
\def\GGG{\mathfrak{G}}
\def\ggg{\mathfrak{g}}

\def\sss{\mathfrak{s}}
\def\S{{\cal S}}
\def\SSS{{\mathfrak S}}
\def\s{{\mathscr{s}}}
\def\Ss{{\mathscr{S}}}

\def\L{\mathscr{L}}
\def\LL{{\cal L}}
\def\tLL{\tilde{\cal L}}
\def\LLL{{\mathfrak L}}
\def\LD{\pounds}
\def\lll{\mathfrak{l}}

\def\iii{\mathfrak{i}}
\def\ii{{\mathrm I}}

\def\JJJ{{\mathfrak J}}

\def\K{{\mathcal K}}
\def\KK{{\mathscr K}}
\def\KKK{{\mathfrak K}}

\def\H{\mathscr{H}}
\def\HH{{\cal H}}
\def\tHH{\tilde{\cal H}}
\def\hhh{{\mathfrak h}}
\def\HHH{{\mathfrak H}}

\def\B{\mathscr{B}}
\def\BB{\bm{B}}
\def\BBB{{\mathfrak B}}
\def\bbb{{\mathfrak b}}

\def\C{\mathbb{C}}
\def\CCC{{\mathfrak C}}
\def\ccc{{\mathfrak c}}


\def\Epsi{\bm{\mathcal E}}
\def\EEE{\mathfrak{E}}
\def\eee{{\mathfrak e}}

\def\I{\mathscr{I}}
\def\tI{\tilde{\mathscr{I}}}
\def\II{{\cal I}}

\def\NN{\mathscr{N}}
\def\N{\mathbb{N}}


\def\R{\mathbb{R}}
\def\Ri{\mathscr{R}}
%\def\Ri{R}
\def\RRR{\mathfrak{R}}
\def\rrr{\mathfrak{r}}

\def\W{{\cal W}}

%\def\vomega{\varpi}  
\def\vomega{{\mathfrak w}}
\def\www{{\mathfrak w}}
\def\WWW{{\mathfrak W}}
%

%\def\kh{\kappa_\hbar}
\def\kh{\hbar G}

\def\tHHH{\widetilde{\HHH}}
\def\tSSS{\widetilde{\SSS}}
\def\twww{\widetilde{\www}}

\def\sc{{\sss\hspace{-.1em}\ccc}}
\def\MMgr{\M_{\hspace{-.1em}G\hspace{-.1em}R}}
\def\MMvac{\M_{\hspace{-.1em}vac}}
\def\LLvac{\LL_{\hspace{-.1em}vac}}
\def\LLschw{\LL_{\hspace{-.1em}\mathrm{Schw}}}
\def\Thetagr{\Theta_{\hspace{-.1em}vac}}
\def\Evac{{\bm{\E}}_{\hspace{-.1em}vac}}

\def\WWWvac{\WWW_{\hspace{-.1em}vac}}
\def\Mvac{M_{\hspace{-.1em}vac}}
\def\WWWgr{\WWW_{\hspace{-.1em}G\hspace{-.1em}R}}
\def\Mgr{M_{\hspace{-.1em}G\hspace{-.1em}R}}
\def\sssvac{\sss_{\hspace{-.1em}vac}}
\def\sssgr{\sss_{\hspace{-.1em}G\hspace{-.1em}R}}
\def\Sgr{S_{\hspace{-.1em}G\hspace{-.1em}R}}

\def\sc{{\sss\hspace{-.1em}\ccc}}
\def\scgr{\sss\hspace{-.1em}\ccc_{\hspace{-.1em}G\hspace{-.1em}R}}
\def\scvac{\sss\hspace{-.1em}\ccc_{\hspace{-.1em}vac}}
\def\sc{{\sss\hspace{-.1em}\ccc}}

\def\Pivac{{\Pi_{\hspace{-.1em}vac}}}

\def\={{\hspace{-.1em}=\hspace{-.1em}}}

\def\LLambda{\bm{\Lambda}}

\def\SO{{\hspace{-.1em}S\hspace{-.1em}O\hspace{-.1em}}}
\def\SU{{\hspace{-.1em}S\hspace{-.1em}U\hspace{-.1em}}}
\def\Uo{{\hspace{-.1em}U\hspace{-.2em}_\textit{1}\hspace{-.1em}}}
\def\Sp{{\hspace{-.1em}s\hspace{-.1em}p\hspace{.1em}}}
\def\SG{{\hspace{-.2em}s\hspace{-.1em}g}}
\def\hSG{{\hspace{-.2em}\widehat{s\hspace{-.1em}g}}}
\def\EM{{\hspace{-.1em}E\hspace{-.1em}M\hspace{-.1em}}}
\def\ch{{\hspace{-.1em}c\hspace{-.1em}h}}
\def\SUO{{\SU\hspace{-.1em}\otimes\SO}}

%\def\Cl{{\textoesch{Cl}\hspace{.1em}}}
\def\Cl{{\mathcal C\hspace{-.1em}l}}

\def\Slash#1{{\not\!#1}}
%\def\Slash#1{{\hspace{-.2em}\slash#1\hspace{.2em}}}
\def\slashD{{D\hspace{-.7em}\slash\hspace{.5em}}}
\def\sss{{\mathfrak s}}
\def\uuu{{\mathfrak u}}

\def\wcs{{\www\hspace{-.1em}\cdot\hspace{-.1em}\Ss}}
\def\wcss{{\(\iota_{{\gamma^{~}_E}_\Sp}\www\)
\hspace{-.1em}\cdot\hspace{-.1em}S}}

\def\gasu{{\(\iota_{{\gamma^{~}_E}_\Sp}\AAA^{~}_\SU\)}}

\def\cP{{\hspace{-.1em}c\hspace{-.05em}\mathrm{P}}}
\def\Poin{{\hspace{-.1em}\mathrm{P}}}
\def\cov{{\hspace{-.1em}cov}}

\def\hpi{{\hat\tau}}
\def\hPi{{\hat\Tau}}

\def\YM{{\mathrm{YM}}}
\def\MYM{{\mathrm{MYM}}}
\def\ym{{\textrm{Y}\hspace{-.1em}\textrm{M}}}

\def\Dc{{\hspace{-.1em}\textrm{D}c}}
\def\hSU{{\widehat\SU}}
\def\muG{{\mu^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\bmuG{{\vec{\mu}^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\gG{{g^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\aG{{a^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\gSM{{g^{~}_{\hspace{-.1em}S\hspace{-.1em}M}}}
\def\aSM{{a^{~}_{\hspace{-.1em}S\hspace{-.1em}M}}}
\def\aExp{{a^{~}_{\hspace{-.1em}E\hspace{-.1em}x\hspace{-.1em}p}}}
\def\BG{{B^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\VG{{V^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\tVG{{\tilde{V}^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\bmBG{{\vec{B}^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\cG{{c^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\cGz{{c^\bare_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\bcG{{\bm{c}^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\cGo{{{c}^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\hcG{{{c}^{~}_{\hspace{-.1em}g\hspace{-.1em}r}}}
\def\DSp{{\D^{~}_{\hspace{-.1em}\Sp}}}
\def\dSp{{\slash{d}^{~}_{\hspace{-.1em}\www}}}
\def\dSg{{\slash{d}^{~}_{\SG}}}
\def\dSgp{{\slash{d}^{+}_{\SG}}}
\def\dSgm{{\slash{d}^{-}_{\SG}}}
\def\dSgpm{{\slash{d}^{\pm}_{\SG}}}
\def\hdSg{{\slash{\hat{d}}^{~}_{\SG}}}
\def\DSpE{{\D_{\hspace{-.1em}\Sp}^E}}


\def\dR{{de\hspace{.1em}Rham }}

\def\cs{{\ccc\hspace{-.1em}\sss}}
\def\SM{{\hspace{-.1em}\setminus\hspace{-.1em}}}

\def\gdg{\bm{g}^{-1}_\SU\hspace{.1em}d\bm{g}_\SU^{~}}

\def\HD{{\widehat{\mathrm{H}}}}
\def\HDi{{\widehat{\mathrm{H}}^{-1}}}
\def\TxT{{2\hspace{-.1em}\times\hspace{-.1em}2}}

\def\sigmav{{{\sigma^{~}_{\hspace{-.1em}v}}}}
\def\sigmag{{{\sigma^{~}_{\hspace{-.1em}g}}}}
\def\tsigmag{{{\tilde\sigma^{~}_{\hspace{-.1em}g}}}}
\def\sigmaeta{{{\sigma^{~}_{\hspace{-.1em}\eta}}}}
\def\sigmath{{{\sigma^{~}_{\hspace{-.1em}\theta}}}}
\def\tsigmath{{{\tilde\sigma^{~}_{\hspace{-.1em}\theta}}}}
\def\sigmaet{{\sigma_{\hspace{-.1em}\eta_\theta}}}

\def\KP{{K\hspace{-.2em}P}}
\def\dr{{\textrm{d\hspace{-.1em}R}}}
\def\theEp{{\theta_\varepsilon}}



%\def\cod{{\rotatebox[origin=c]{180}{$\delta$}}}
\def\cod{{\bar{d}}}
%\def\cod{{\hspace{.1em}\reflectbox{\textit{d}}\hspace{-.1em}}}
%\def\cod{{\delta}}

\def\<{{\langle}}
\def\>{{\rangle}}

\def\norm{{\textit{norm}}}

\def\Land{{~\land~}}


\def\({\left(}
\def\){\right)}

\def\tMg{{\tilde{\M}_{\hspace{-.1em}\theta}}}
\def\TtMg{{{T\hspace{-.2em}\tilde{\M}}_{\hspace{-.1em}\theta}}}
\def\TstMg{{{T^*\hspace{-.2em}\tilde{\M}}_{\hspace{-.1em}\theta}}}
\def\TtM{{{T\hspace{-.2em}\tilde{\M}}}}
\def\TstM{{{T^*\hspace{-.2em}\tilde{\M}}}}

%\def\teee{\tilde{\mathfrak{e}}}
%\def\teee{d\hp}
%\def\teee{\ppp}
\def\teee{\xxx}
\def\heee{d\hr}
\def\teepp{\tilde{\mathfrak{e}}_\mathfrak{p}}

\def\CMm{{C\hspace{-.15em}M}}
\def\CM{{C\hspace{-.15em}\tilde{M}}}
\def\MC{{\tilde{M}\hspace{-.15em}C}}
%\def\pF{{p\hspace{-.1em}F}}
\def\pF{{F\hspace{-.1em}T}}
\def\FT{{C\hspace{-.2em}F}}

\def\FTg{{C\hspace{-.2em}F_\theta}}
\def\CMmg{{C\hspace{-.15em}M_\theta}}
\def\CMg{{C\hspace{-.15em}\tilde{M}_\theta}}
\def\MCg{{\tilde{M}\hspace{-.15em}C_\theta}}

\def\tp{\tilde{p}}
\def\tx{\tilde{x}}
\def\tr{\tilde{r}}

\def\hp{\hat{p}}
\def\hx{\hat{\xi}}
\def\hr{\hat{r}}

\def\bp{\bm{p}}
\def\bq{\bm{q}}
\def\bx{\bm{x}}
\def\br{\bm{r}}

\def\btp{\tilde{\bp}}
\def\btx{\tilde{\bx}}
\def\btr{\tilde{\br}}

\def\vvvp{{\VVV\hspace{-.1em}_\mathrm{p}\hspace{-.15em}}}
\def\tvvv{{\tilde\vvv}}
\def\tvvvp{{\tvvv\hspace{-.1em}_\mathrm{p}}}

\def\deteps{{\mathrm{det}\left[\E\right]}}
\def\deteta{{\mathrm{det}\left[\bm{\eta}^{~}_\theta\right]}}

\def\fdl{{\bm{[}}}
\def\fdr{{\bm{]}}}

\def\E{{\mathcal E}}
\def\Etr{{\mathcal E}\hspace{-.1em}3}

\def\fsch{f_{\textrm{S}}}
\def\sinvt{\sin{\hspace{-.15em}\vartheta}}
\def\sintvt{\sin^2{\hspace{-.15em}\vartheta}}

\def\gmt{{g_\mu\hspace{-.2em}-\hspace{-.1em}2}}

\def\pb{{\bar{q}}}
