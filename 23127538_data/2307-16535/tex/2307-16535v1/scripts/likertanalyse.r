library(likert)

setwd('C:/Users/ryank/Documents/papers/security_cards_paper/scripts')

# Summer School (Workshop 2)
ws2 <- read.csv('workshop2likertdata.csv')

l <- likert(ID~., ws2)

# l <- likert(Question~., ws2)
# 
png("ws2likertplot.png", height=720, width=1080)
plot(l, ordered=F, group.order = names(l[2:9]))
dev.off()

# png("ws2likertplot.png", height=720, width=1080)
# p <- likert.bar.plot(l, center = (l$nlevels - 1)/2 + 1)
# dev.off()