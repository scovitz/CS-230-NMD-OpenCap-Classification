library("irr")

input <- file('stdin', 'r')
data <- read.delim(input, sep=',', header=FALSE)
res <- icc(data, model = "twoway", type = "agreement", unit = "single")

cat(paste0('icc,', res$value, '\n'))
cat(paste0('lbound,', res$lbound, '\n'))
cat(paste0('ubound,', res$ubound, '\n'))
cat(paste0('conf_level,', res$conf.level, '\n'))
cat(paste0('Fvalue,', res$Fvalue, '\n'))
cat(paste0('df1,', res$df1, '\n'))
cat(paste0('df2,', res$df2, '\n'))
cat(paste0('p,', res$p.value, '\n'))
cat(paste0('r0,', res$r0, '\n'))

