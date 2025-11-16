import coffee
import research
from research import paper

if coffee.empty():
    coffee.make()
else:
    coffee.drink()
    for paper in papers:
        info = paper.read()
        literatureReview.append(info)
    research.writePaper(literatureReview)



