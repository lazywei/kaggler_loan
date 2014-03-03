train.cleaner <- function(train.data) {

  fill.missing <- function(x, feature.means) {
    na.ind <- which(is.na(x))
    x[na.ind] <- feature.means[na.ind]
    return(x)
  }

  feature.means <- colMeans(train.data[,-ncol(train.data)], na.rm=TRUE)

  cleaner <- function(df) {
    incomplete.ind <- which(!complete.cases(df))
    df[incomplete.ind,] <- apply(df[incomplete.ind,], 1, fill.missing, feature.means=feature.means)

    return(df)
  }

  return(cleaner)
}
