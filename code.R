install.packages("klaR")
install.packages("nnet")
install.packages("liquidSVM")
install.packages("rpart")
install.packages("randomForest")
install.packages("class")

#### read data ####
d<-read.table("C:\\Users\\purka\\Downloads\\page-blocks.data\\page-blocks.data")
d$V11<-as.factor(d$V11)

str(d)

#### training set index ####
training<-c(sample(which(d$V11==1), round(0.5*length(which(d$V11==1)))), 
            sample(which(d$V11==2), round(0.5*length(which(d$V11==2)))), 
            sample(which(d$V11==3), round(0.5*length(which(d$V11==3)))), 
            sample(which(d$V11==4), round(0.5*length(which(d$V11==4)))), 
            sample(which(d$V11==5), round(0.5*length(which(d$V11==5)))))
            

#### lda ####
lda.model<-MASS::lda(V11~., 
               data=d, 
               subset=training)
caret::confusionMatrix(predict(lda.model, 
                               newdata=d[-training, ])$class, 
                       d[-training, 11])

#### qda ####
qda.model<-MASS::qda(V11~., 
                     data=d, 
                     subset=training)
caret::confusionMatrix(predict(qda.model, 
                               newdata=d[-training, ])$class, 
                       d[-training, 11])



#### rda ####
alpha<-(1:9)/10

rda.error<-function(alpha){
  rda.model<-klaR::rda(V11~., 
                       d, 
                       subset=training, 
                       lambda = alpha, 
                       crossval=TRUE, 
                       fold=5)
  
  return(confusionMatrix(predict(rda.model, d[training,-11])$class, 
                         d[training,11]
  )$overall[1])
}

results<-sapply(alpha, rda.error)

alpha.opt<-alpha[max(which(results==max(results)))]

rda.model<-klaR::rda(V11~., 
                     d, 
                     lambda = alpha.opt, 
                     crossval=TRUE, 
                     fold=5)

caret::confusionMatrix(predict(rda.model,
                               newdata=d[-training, -11])$class, 
                       d[-training, 11])
#### logistic model ####
log.model<-nnet::multinom(V11~., 
                          data=d[training,])
caret::confusionMatrix(predict(log.model, 
                               newdata = d[-training, -11]), 
                       d[-training, 11])

#### svm ####
svm.ovo.model<-liquidSVM::mcSVM(V11~., 
                                d[training, ], 
                                mc_type="AvA_hinge", 
                                max_gamma=3025)
caret::confusionMatrix(predict(svm.ovo.model, 
                               newdata = d[-training, -11]), 
                       d[-training, 11])

svm.ova.model<-liquidSVM::mcSVM(V11~., 
                                d[training, ], 
                                mc_type="OvA_hinge", 
                                max_gamma=3025)
caret::confusionMatrix(predict(svm.ova.model, 
                               newdata = d[-training, -11]), 
                       d[-training, 11])

#### tree ####
## unpruned
helper.classify<-function(x){
  c<-ifelse(x[1]==max(x), 1,
        ifelse(x[2]==max(x), 2, 
               ifelse(x[3]==max(x), 3, 
                      ifelse(x[4]==max(x), 4, 5))))
  return(c)
}
tree.model <- rpart::rpart(V11~., 
                           data=d, 
                           subset= training, 
                           method="class", 
                           control=rpart::rpart.control(misplit=50, cp=0.01))

caret::confusionMatrix(as.factor(apply(predict(tree.model, 
                                               newdata = d[-training, -11]), 1, helper.classify)),
                       d[-training, 11])

## pruned
pruned.tree.model<-rpart::prune(tree.model, cp=tree.model$cptable[which.min(tree.model$cptable[,"xerror"]),"CP"])
caret::confusionMatrix(as.factor(apply(predict(pruned.tree.model, 
                                               newdata = d[-training, -11]), 1, helper.classify)),
                       d[-training, 11])



#### random forest ####
library(randomForest)
splits.error.rf<-function(k){
  rf.model<-randomForest(V11~., 
                         data=d, 
                         subset=training, 
                         mtry=k)
  return(caret::confusionMatrix(predict(rf.model, 
                                 newdata=d[-training, -11]), 
                         d[-training, 11])$overall[1])
}
set.seed(1234)
k.opt<-(3:8)[which.max(sapply(3:8, splits.error.rf))]

rf.model<-randomForest(V11~., 
                       data=d,
                       subset=training, 
                       importance=TRUE)
caret::confusionMatrix(predict(rf.model, 
                               newdata=d[-training, -11]), 
                       d[-training, 11]) 

#### kNN ####
num.neigh.knn<-function(k){
  return(
    caret::confusionMatrix(class::knn(d[training,], d[-training, ], d[training, 11], k), 
                           d[-training, 11])$overall[1]
  )
}
k.opt<-c(1, seq(5, 20, 5))[which.max(sapply(c(1, seq(5, 20, 5)), num.neigh.knn))]

knn.model<-class::knn(d[training,], d[-training, ], d[training, 11], k.opt)
caret::confusionMatrix(class::knn(d[training,], d[-training, ], d[training, 11], k.opt), 
                       d[-training, 11])
  