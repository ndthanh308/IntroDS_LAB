# setup -------------------------------------------------------------------
rm(list=ls())
library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# setup data --------------------------------------------------------------
data(iris)
dat = as.matrix(iris[,1:4])
lab = as.integer(as.factor(iris[,5]))

alpha = 0.25
col_points = rep(0, length(lab))
col_shadow = rep(0, length(lab))
for (i in 1:length(lab)){
  if (lab[i]==1){
    col_points[i] = rgb(0,0,0,1)
    col_shadow[i] = rgb(0,0,0,alpha)
  } else if (lab[i]==2){
    col_points[i] = rgb(1,0,0,1)
    col_shadow[i] = rgb(1,0,0,alpha)
  } else if (lab[i]==3){
    col_points[i] = rgb(0,0,1,1)
    col_shadow[i] = rgb(0,0,1,alpha)
  }
}


# PCA ---------------------------------------------------------------------
# method 1 : exact PCA
eigcov  = eigen(cov(dat))
proj1   = eigcov$vectors[,1:2]
method1 = dat%*%proj1
method1 = as.matrix(scale(method1, center=TRUE, scale=FALSE))

# method 2 : SVD without centering
svddat  = base::svd(dat)
proj2   = svddat$v[,1:2]
method2 = dat%*%proj2
method2 = as.matrix(scale(method2, center=TRUE, scale=FALSE))



# Visualization -----------------------------------------------------------
# procrustes matching
QQ = pracma::procrustes(method1, method2)$Q
method3 = as.matrix(scale(method2%*%QQ, center=TRUE, scale=FALSE))

# naive
par(mfrow=c(1,3), pty="s")
plot(method1, col=lab, pch=19, main = "exact")
plot(method2, col=lab, pch=19, main = "SVD")
plot(method3, col=lab, pch=19, main = "SVD+rotate")





# save the figure : base --------------------------------------------------
# parameters
par_cexmain = 2.5
par_cexlab  = 2
par_cexpts  = 0.85


# Exact PCA
graphics.off()
png(file="code1_1exactPCA.png", width=500, height=600)
plot(method1, col=col_points, pch=19, main="(a) Exact PCA", xlab="Dimension 1", ylab="Dimension 2",
     cex=par_cexpts, cex.main=par_cexmain, cex.lab=par_cexlab, xlim=c(-3.5, 4.2), ylim=c(-1.5, 1.5))
dev.off()

# SVD Rotated
graphics.off()
png(file="code1_2rotateSVD.png", width=500, height=600)
plot(method3, col=col_points, pch=19, main="(b) SVD without Centering", xlab="Dimension 1", ylab="Dimension 2",
     cex=par_cexpts, cex.main=par_cexmain, cex.lab=par_cexlab, xlim=c(-3.5, 4.2), ylim=c(-1.5, 1.5))
dev.off()

# Difference
graphics.off()
png(file="code1_3difference.png", width=500, height=600)
diff_cex = 1.1
plot(method1, col=col_shadow, pch=10, main="(c) Difference", xlab="Dimension 1", ylab="Dimension 2",
     cex=diff_cex, cex.main=par_cexmain, cex.lab=par_cexlab, xlim=c(-3.5, 4.2), ylim=c(-1.5, 1.5))
points(method3, col=col_shadow, pch=10, cex=diff_cex)
for (i in 1:length(lab)){
  # extract coordinates
  pt1 = method1[i,]
  pt2 = method3[i,]

  # specify x{start,end} and y{start, end}
  xx = c(pt1[1],pt2[1])
  yy = c(pt1[2],pt2[2])
  lines(xx,yy,lwd=4,col=col_points[i],lend=0)
}
dev.off()


# save the figure : all three ---------------------------------------------
# setup
graphics.off()
par_cexmain = 2.25
par_cexlab  = 1.5
par_cexpts  = 1.1

# draw
png(file="code1_all.png", width=800, height=600)
par(mfrow=c(1,3))
plot(method1, col=col_points, pch=19, main="Exact PCA", xlab="Dimension 1", ylab="Dimension 2",
     cex=par_cexpts, cex.main=par_cexmain, cex.lab=par_cexlab, xlim=c(-3.5, 4.2), ylim=c(-1.5, 1.5))
plot(method3, col=col_points, pch=19, main="SVD without Centering", xlab="Dimension 1", ylab="Dimension 2",
     cex=par_cexpts, cex.main=par_cexmain, cex.lab=par_cexlab, xlim=c(-3.5, 4.2), ylim=c(-1.5, 1.5))
diff_cex = 1.1
plot(method1, col=col_shadow, pch=10, main="Difference", xlab="Dimension 1", ylab="Dimension 2",
     cex=diff_cex, cex.main=par_cexmain, cex.lab=par_cexlab, xlim=c(-3.5, 4.2), ylim=c(-1.5, 1.5))
points(method3, col=col_shadow, pch=10, cex=diff_cex)
for (i in 1:length(lab)){
  # extract coordinates
  pt1 = method1[i,]
  pt2 = method3[i,]
  
  # specify x{start,end} and y{start, end}
  xx = c(pt1[1],pt2[1])
  yy = c(pt1[2],pt2[2])
  lines(xx,yy,lwd=1,col=col_points[i],lend=0)
}
dev.off()