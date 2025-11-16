j=revised # ieee
platex $j
bibtex $j
platex $j
platex $j
dvipdfmx ${j}
evince --fullscreen ${j}.pdf

