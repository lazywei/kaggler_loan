require('e1071')

data.train <- read.csv("../cleaned_data/cleaned_train.csv", header=T, colClasses=c("numeric"))
data.test  <- read.csv("../cleaned_data/cleaned_test.csv", header=T, colClasses=c("numeric"))

# Use features selected by FSelector
data.train <- data.train[, c('f148', 'f147', 'f146', 'f663', 'f653', 'f180', 'f170', 'f145', 'f640', 'f726', 'loss')]
data.test  <- data.test[, c('f148', 'f147', 'f146', 'f663', 'f653', 'f180', 'f170', 'f145', 'f640', 'f726')]

default.ind <- which(data.train$loss != 0)
solvent.ind <- which(data.train$loss == 0)
# Resample solvent index to balance data
solvent.ind <- sample(solvent.ind, length(default.ind))

train.default <- data.train[default.ind,]
train.solvent <- data.train[solvent.ind,]

# Set all default loss to 1, then use SVM for binary classification
train.default$loss <- 1
balanced.data      <- rbind(train.default, train.solvent)
balanced.data$loss <- factor(balanced.data$loss)

classifier       <- svm(loss ~ ., data=balanced.data, type='C-classification')
pred             <- predict(classifier, data.test)
pred             <- as.numeric(levels(pred))[pred]
pred.default.ind <- which(pred == 1)

# Use original default data to train SVM
train.default          <- data.train[default.ind,]
svm.model              <- svm(loss ~ ., data=train.default)
pred[pred.default.ind] <- predict(svm.model, data.test[pred.default.ind,])

# Make submission
sub     <- read.csv("../orig/sampleSubmission.csv")
sub[,2] <- pred
write.csv(sub, "sub.csv", row.names=F)

save.image()
