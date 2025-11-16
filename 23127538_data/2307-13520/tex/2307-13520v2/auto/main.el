;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper" "10pt" "twocolumn")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("hyperref" "colorlinks=true" "linkcolor=blue" "citecolor=blue" "urlcolor=blue") ("cleveref" "capitalise" "noabbrev") ("babel" "UKenglish") ("geometry" "left=2cm" "right=2cm" "top=2cm" "bottom=2.5cm") ("stix2" "notextcomp") ("titlesec" "small")))
   (TeX-run-style-hooks
    "latex2e"
    "supplementary_material"
    "article"
    "art10"
    "placeins"
    "etoolbox"
    "booktabs"
    "chemformula"
    "hyperref"
    "caption"
    "subcaption"
    "cleveref"
    "authblk"
    "babel"
    "graphicx"
    "parskip"
    "geometry"
    "stix2"
    "titlesec"
    "doi")
   (LaTeX-add-labels
    "sec:data-methods"
    "sec:met-inputs"
    "sec:modelling"
    "sec:duals"
    "sec:system-defining"
    "sec:weather-regimes"
    "sec:clustering"
    "sec:load-shedding"
    "sec:results"
    "tab:approaches"
    "fig:approaches"
    "sec:duration-periods"
    "fig:event-overview"
    "sec:origins-stress"
    "fig:key-metrics"
    "sec:system-interactions"
    "fig:event-example"
    "sec:comparison-traditional"
    "fig:sys-def-weather"
    "fig:regimes"
    "fig:evolution-regimes"
    "fig:NAO_total_costs"
    "sec:validation"
    "sec:discussion-conclusions")
   (LaTeX-add-bibliographies
    "references.bib"))
 :latex)

