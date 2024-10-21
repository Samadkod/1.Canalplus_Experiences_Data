# Chargement des bibliothèques (packages nécessaires)
library(dplyr)
library(Matrix)
library(glmnet)
library(nnet)
library(ggplot2)
library(lattice)
library(caret)
library(randomForest)
library(tidyverse)
library(data.table)
library(kernlab)
library(xgboost)
library(caret)
library(pROC)
library(e1071)

# CHARGEMENT DES DONNEES


# Importation du fichier : train "amf_data.csv"
library(readr)
amf_data <- read_csv("~/forecast/amf_data.csv", show_col_types = FALSE)
s <- spec(amf_data)
print(s)
cols_condense(s)
summary(amf_data)

# Supression du type de trader "Mix", car le trader mix utilise souvent plusieurs Approches de trading Plutôt que de se concentrer
# Sur une seule méthode (il utilise à la fois du HFT et Non HFT). Le type de trader "Mix" n'est pas contrôlé.
# Il pourra donc fausser les résultats des analyses, d'où l'intérêt de le supprimer
amf_data <- na.omit(amf_data)
amf_data = amf_data[!(amf_data$type=='MIX'),]

# Création dummy variable : variable binaire pour la variable "type_trader"
amf_data$type_trader = ifelse(amf_data$type=='HFT', 1, 0)
str(amf_data)

# Supprimer la variable "trader" de dataset
amf_data <- subset(amf_data, select = -trader)

# Création d'une nouvelle base de données intermédiaire "intermed_data"
intermed_data = amf_data

# Affichage des premières lignes du jeu de données
head(intermed_data)

# EXPLORATION DES DONNEES


# POUR Obtenir une vue d'ensemble des données
summary(intermed_data)

# Afficage des fréquences de chaque classe
table(intermed_data$type)

# Sélectionner les variables à utiliser dans le modèle
vars_to_use <- c("type_trader","otr", "ocr", "omr", "min_time_two_events", "mean_time_two_events", "max_time_two_events", "min_lifetime_cancel", "mean_lifetime_cancel", "max_lifetime_cancel", "nbtradevenuemic", "maxnbtradesbysecond", "meannbtradesbysecond", "nbsecondwithatleatonetrade")

# Affichage de la corrélation entre les variables
cor(intermed_data[, vars_to_use])

# Diviser la fenêtre graphique en 4 colonnes et 3 lignes
par(mfrow=c(3,4))

# Visualiser la distribution de chaque variable
hist(intermed_data$otr)
hist(intermed_data$ocr)
hist(intermed_data$omr)
hist(intermed_data$min_time_two_events)
hist(intermed_data$mean_time_two_events)
hist(intermed_data$max_time_two_events)
hist(intermed_data$min_lifetime_cancel)
hist(intermed_data$mean_lifetime_cancel)
hist(intermed_data$max_lifetime_cancel)
hist(intermed_data$nbtradevenuemic)
hist(intermed_data$maxnbtradesbysecond)
hist(intermed_data$meannbtradesbysecond)
hist(intermed_data$nbsecondwithatleatonetrade)

# PRETRAITEMENT DE DONNEES


# La mise en log des variables pour limiter la valeur des fréquences
intermed_data$log_otr <- log(intermed_data$otr)
intermed_data$log_ocr <- log(intermed_data$ocr)
intermed_data$log_omr <- log(intermed_data$omr)
intermed_data$log_nbtradevenuemic <- log(intermed_data$nbtradevenuemic)
intermed_data$log_maxnbtradesbysecond <- log(intermed_data$maxnbtradesbysecond)
intermed_data$log_meannbtradesbysecond <- log(intermed_data$meannbtradesbysecond)

# Création d'une nouvelle dataset (mydata) + Suppression des variables originales
mydata <- intermed_data[, -c(2:14)]

# Afficher les premières lignes du jeu de données après prétraitement
head(mydata)


# CONSTRUCTION DES MODELES DE MACHINE LEARNING


# Séparation des données en données d'entraînement (Non HFT) et données de test (HFT)
set.seed(123)
train_index <- createDataPartition(mydata$type_trader, p = .8, list = FALSE, times = 1)
train_data <- mydata[train_index, ]
test_data <- mydata[-train_index, ]

summary(train_data)
summary(test_data)

# Vérification de structure et de dimension
str(train_data)
str(test_data)
dim(test_data)
dim(train_data)
table(train_data$type_trader)
table(test_data$type_trader)

# Définissons les modèles à tester
# Modèle de regression logistique (glm)
# Modèle "rpart" : Classification and Regression Tree (CART) via Recursive Partitioning.
# Modèle de forêt aléatoire "rf"
# Modèle "gbm" : Gradient Boosting Machine.
# Modèle "svmRadial" : Support Vector Machines (SVM) avec noyau radial.
# Modèle de réseau de neurones (nnet)

models <- list(
  "glm" = glm(type_trader ~ ., data = train_data, family = "binomial"),
  "rf" = randomForest::randomForest(type_trader ~ ., data = train_data, mtry = sqrt(ncol(train_data)), importance = TRUE, na.action = na.omit),
  "svmRadial" = kernlab::ksvm(type_trader ~ ., data = train_data, kernel = "rbfdot")
)

print(models)

summary(models$glm)
summary(models$rf)
summary(models$svmRadial)

plot(models$glm)
plot(models$rf)

# Evaluation des performances du modèle
# Prédiction sur l'ensemble de test
# Évaluation des performances

# Calcul de la prédiction pour chaque modèle
glm_pred <- predict(models$glm, newdata = test_data, type = "response")
print(glm_pred)
rf_pred <- predict(models$rf, newdata = test_data)
print(rf_pred)
svmRadial_pred <- predict(models$svmRadial, newdata = test_data)
print(svmRadial_pred)

# Visualiser les prédictions de chaque modèle pour les classes de trader humain (non HFT) et ordi (HFT)
# "#2c7bb6" = couleur bleue si humain (Non HFT) et,
# "#d7191c" = couleur rouge si ordinateur (HFT)
for (i in 1:length(models)) {
  if (names(models)[i] == "gbm") {
    pred <- predict(models[[i]], as.matrix(test_data[, vars_to_use_log]))
  } else {
    pred <- predict(models[[i]], test_data)
  }
  levels(test_data$type_trader) <- levels(factor(pred, levels = levels(test_data$type_trader)))
  table(test_data$type_trader, pred)
  barplot(table(test_data$type_trader, pred), main = names(models)[i], col = c("#2c7bb6", "#d7191c"))
}

# Calcul de la précision pour chaque modèle
glm_acc <- mean((glm_pred > 0.5) == test_data$type_trader)
rf_acc <- mean((rf_pred > 0.5) == test_data$type_trader)
svmRadial_acc <- mean((svmRadial_pred > 0.5) == test_data$type_trader)
acc <- c(glm_acc, rf_acc, svmRadial_acc)
names(acc) <- c("glm", "rf", "svmRadial")
acc

# Matrix de confusion
conf_matrix_glm = table(glm_pred)
view(conf_matrix_glm)
conf_matrix_rf = table(rf_pred)
view(conf_matrix_rf)
conf_matrix_svm = table(svmRadial_pred)
view(conf_matrix_svm)

# On peut également visualiser les prédictions avec les matrix de confusion
plot(conf_matrix_glm, col = c("#2c7bb6", "#d7191c"))
plot(conf_matrix_rf, col = c("#2c7bb6", "#d7191c"))
plot(conf_matrix_svm, col = c("#2c7bb6", "#d7191c"))

# Utiliser le modèle de forêt aléatoire pour prédire qui opère derrière le trade
# Faire des prédictions pour les données de test
best_model <- models$rf
pred_rf <- predict(best_model, newdata = test_data, type = "response")
print(best_model)
print(pred_rf)

# Créer un dataframe de sortie
output <- data.frame(Id = test_data$type_trader, Prediction = pred)
print(output)


# -----------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------








