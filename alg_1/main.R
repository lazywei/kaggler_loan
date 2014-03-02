source("../common/train_cleaner.R")

# data.result <- read.csv("result.csv", header=T, colClasses=c("numeric"))

paramTable <- c(70, 40, 80)

for (i in 1:1) {
	k_mean.nc <- paramTable[1]
	k_mean.iter <- paramTable[2]
	k_mean.max_iter <- paramTable[3]
	print(paramTable)

	data.train <- read.csv("../orig/train_v2.csv", header=T, colClasses=c("numeric"))
	data.test  <- read.csv("../orig/test_v2.csv", header=T, colClasses=c("numeric"))
	print("--- read file done ---")

	numCol <- ncol(data.train)
	cleaner <- train.cleaner(data.train, k_mean.n_centers=k_mean.nc, k_mean.iter=k_mean.iter, k_mean.max_iter=k_mean.max_iter)
	print("--- train cleaner done ---")

	data.train[,-numCol] <- cleaner(data.train[,-numCol])
	print("--- clean data.train done ---")

	data.test <- cleaner(data.test)
	print("--- clean data.test done ---")

	write.table(data.train, "for_train.csv", sep=",", row.names=F, col.names=F)
	write.table(data.test, "for_test.csv", sep=",", row.names=F, col.names=F)

	system("python ../python_code/loan_quantile_2.py for_train.csv for_test.csv out.csv")

	ys <- read.csv("out_test.csv", header=F, colClasses=c("numeric"))
	# MAE <- sum(abs(ys - data.test[,numCol])) / nrow(ys)
	# data.result <- rbind(data.result, c(k_mean.nc, k_mean.iter, k_mean.max_iter, MAE))
	# write.csv(data.result, "result.csv", row.names=F) 
}

save.image()
