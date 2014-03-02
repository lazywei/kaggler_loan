# Parameters:
# data_train 應該以這種方式讀入
# data_train <- read.csv("orig/train_30k.csv", header=T, colClasses=c("numeric"))
# 請參考 Dropbox/TPE_Kaggler/load_prediction/data/README
#
# Usage:
# source("./train_cleaner.R")
# cleaner <- train.cleaner(data_train)
# cleaner(default_data, is.solvent=F)
# cleaner(solvent_data, is.solvent=T)

train.cleaner <- function(data_train, k_mean.iter = 30, k_mean.n_centers = 200, k_mean.max_iter = 50) {

  # --- pre-define functions ---

  # countNA will return the number of NA in a given vector X
  # If logic=T, it work as any(is.na(X)) function
  countNA <- function(X,logic=F) {
    count <- sum(is.na(X))
    if (logic) {
      count <- as.logical(count)
    }
    return(count)
  }

  # getMode 回傳給定向量的眾數
  getMode <- function(x) {
    ux <- unique(x)
    return(ux[which.max(tabulate(match(x, ux)))])
  }

  # getCluster 回傳給定向量 x 所屬的 cluster
  getCluster <- function(x, centers) {
    if (any(is.na(x))) {
      ind_na <- which(is.na(x))
      centers <- centers[,-ind_na]
      x <- x[-ind_na]
    }
    colNum <- ncol(centers)
    rowNum <- nrow(centers)
    temp <- t(matrix(x, nrow=colNum, ncol=rowNum))

    distToCenters <- rowSums(sqrt((temp - centers)^2))

    return(which.min(distToCenters))
  }

  # fillData 填入所屬 cluster 的眾數
  fillData <- function(x, featureTable, centers) {
    whichCluster <- getCluster(x, centers)

    ind_na <- which(is.na(x))
    x[ind_na] <- featureTable[whichCluster, ind_na]
    return(x)
  }

  # ----- End of definition ----

  dataNum    <- dim(data_train)[1]
  featureNum <- dim(data_train)[2] - 1
  X_train    <- data_train[,1:featureNum]
  y_train    <- data_train[,featureNum+1]


  # 1. 把有 NA 和沒 NA 的資料分開來 -> dirty & clean

  # Find the index of observations which have missing value(s).
  ind <- apply(X_train,1,countNA,logic=T)

  # Seperate data into two group, clean and dirty one.
  X_dirty <- X_train[ind,]
  X_clean <- X_train[!ind,]
  y_dirty <- y_train[ind]
  y_clean <- y_train[!ind]

  # Remove data not needed to save memory space.
  rm(list=c('X_train', 'y_train', 'data_train', 'ind', 'dataNum'))

  # ------- End of 1. -------


  # 2. 把 dirty & clean 的資料再依據 loss 是不是 0 分成 default & solvent

  # Futher seperate data into default group and solvent group.
  ind_clean_default <- which(y_clean!=0)
  X_clean_default   <- X_clean[ind_clean_default,]
  X_clean_solvent   <- X_clean[-ind_clean_default,]
  y_clean_default   <- y_clean[ind_clean_default]
  y_clean_solvent   <- y_clean[-ind_clean_default]

  ind_dirty_default <- which(y_dirty!=0)
  X_dirty_default   <- X_dirty[ind_dirty_default,]
  X_dirty_solvent   <- X_dirty[-ind_dirty_default,]
  y_dirty_default   <- y_dirty[ind_dirty_default]
  y_dirty_solvent   <- y_dirty[-ind_dirty_default]

  # feature_names <- colnames(X_clean)
  rm(list=c('X_clean','y_clean','X_dirty','y_dirty', 'ind_clean_default', 'ind_dirty_default'))

  # ------- End of 2. -------


  # 3. 用 clean+default 的資料去跑 k-means，得到 k 個 centers

  centers <- matrix(0, ncol=featureNum, nrow=0)

  for (i in 1:k_mean.iter) {
    centers <- rbind(centers,
                     kmeans(X_clean_default, algorithm="Lloyd", centers=k_mean.n_centers,iter.max=k_mean.max_iter)$centers)
  }

  # 對所有的 centers 再做一次 k-means
  centers <- kmeans(centers, algorithm="Lloyd", center=k_mean.n_centers, iter.max=k_mean.max_iter)$centers
  centers <- rbind(centers,
                   colMeans(X_clean_solvent))

  # ------- End of 3. -------

  # 4. 把所有 clean 的 data 都歸進某個 cluster

  default.cluster.ind <- apply(X_clean_default, 1, getCluster, centers=centers)
  solvent.cluster.ind <- apply(X_clean_solvent, 1, getCluster, centers=centers)

  # ------- End of 4. -------

  # 5. 計算每個 cluster 的眾數
  #    i-th row 代表第 i 個 cluster 的眾數 feature
  #    把 clean+solvent 的資料另外當成一群（所以現在應該有 k+1 個 clusters，k 個是 default、1 個是 solvent）
  #    solvent 是第一個 cluster

  featureTable <- matrix(0, nrow=k_mean.n_centers+1, ncol=featureNum)
  for (i in 1:(k_mean.n_centers+1)) {
    tmp.default <- X_clean_default[which(default.cluster.ind==i),]
    tmp.solvent <- X_clean_solvent[which(solvent.cluster.ind==i),]
    ith.cluster.data <- rbind(tmp.default, tmp.solvent)
    if (nrow(ith.cluster.data) > 0) {
      featureTable[i,] <- apply(ith.cluster.data, 2, getMode)
    }
  }

  # ------- End of 5. -------

  # 6. 把 dirty data 分進他們所屬的 cluster 並填資料

  cleaner <- function(df) {
    ind      <- which(!complete.cases(df))
    if (length(ind) > 0 ) {
      df[ind,] <- apply(df[ind,], 1, fillData, featureTable=featureTable, centers=centers)
    }
    return(df)
  }

  # ------- End of 6. -------

  return(cleaner)
}
