library(rjags)
set.seed(26)
.pardefault <- par(no.readonly = TRUE)

read_mnist56 <- function(path_or_url, shuffle = FALSE, n.take = nrow(data)) {
  data = read.table(path_or_url, sep=',', header=TRUE)
  if (shuffle) {
    data = data[sample(nrow(data)),]
  }
  x = data.matrix(data[,-ncol(data)])
  x[x > 0] = 1
  y = data[, ncol(data)]
  if (!is.null(n.take)) {
    n.take = min(n.take, length(x))
    x = x[1:n.take,]
    y = y[1:n.take]
  }
  data_list = list(x=x, y=y)
  return(data_list)
}

data_train = read_mnist56(url("https://www.dropbox.com/s/l7uppxi1wvfj45z/MNIST56_train.csv?dl=1"), shuffle = TRUE, n.take=500)
data_test = read_mnist56(url("https://www.dropbox.com/s/399gkdk9bhqvz86/MNIST56_test.csv?dl=1"))

mod_string = "model {
  for (i in 1:length(y)) {
    y[i] ~ dbern(p[i])
    logit(p[i]) = z[i,2] - z[i,1]
    z[i,1:2] = w %*% x[i,]
  }
  for (i_output in 1:2) {
    for (j_input in 1:25) {
      w[i_output, j_input] ~ dbern(0.5)
    }
  }
}"
mod = jags.model(textConnection(mod_string), data=data_train, n.chains=3)
mod_sim = coda.samples(model=mod, variable.names='w', n.iter=200)
mod_csim = as.mcmc(do.call(rbind, mod_sim))

# plot(mod_sim)
# effectiveSize(mod_csim)

as.matrix.2x25 <- function(w) {
  w = as.numeric(w > 0.5)
  w = matrix(w, nrow=2, ncol=25, byrow=FALSE)
  return(w)
}

calc_accuracy <- function(w, data) {
  z = data$x %*% t(w)
  py = 1 / (1 + exp(z[,1] -z[,2]))
  y_predicted = as.numeric(py > 0.5)
  confusion_matrix = table(data$y, y_predicted)
  accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix)
  return(accuracy)
}

w = as.matrix.2x25(colMeans(mod_csim))
accuracy_train = calc_accuracy(w, data_train)
accuracy_test = calc_accuracy(w, data_test)
cat("MCMC accuracy train = ", accuracy_train, ", test = ", accuracy_test)

plot_convergence <- function(n.step = 10, at_epoch=TRUE) {
  chain_colors = c('red', 'green', 'blue')
  for (chain_id in 1:3) {
    draws_chain = mod_sim[[chain_id]]
    draws_chain = draws_chain[, startsWith(colnames(draws_chain), 'w')]
    iteration = seq(from=n.step, to=nrow(mod_sim[[1]]), by=n.step)
    epoch_accuracy_func  <- function(epoch) {
      if (at_epoch || epoch == 1) {
        w = draws_chain[epoch,]
      } else if (!at_epoch) {
        w = colMeans(draws_chain[1:epoch,])
      }
      w = as.matrix.2x25(w)
      return(calc_accuracy(w, data_train))
    }
    accuracy = lapply(iteration, epoch_accuracy_func)
    if (chain_id == 1) {
      if (at_epoch) {
        title = "at epoch"
      } else {
        title = "mean(W1,..,Wi)"
      }
      title = paste("Convergence diagnostic", title)
      plot(iteration, accuracy, type='n', ylim=c(0.5, 1), main=title)
    }
    points(jitter(iteration, amount = 0.5), accuracy, col=chain_colors[chain_id])
  }
  legend('bottomright', legend=c('chain 1', 'chain 2', 'chain 3'), col=chain_colors, fill=chain_colors)
}
par(mfrow=c(2, 1))
plot_convergence(n.step=1, at_epoch=TRUE)
plot_convergence(n.step=1, at_epoch=FALSE)

check_residuals <- function() {
  z = data_train$x %*% t(w)
  py = 1 / (1 + exp(z[,1] -z[,2]))
  resid = data_train$y - py
  par(.pardefault)
  colors.digit = c('blue', 'green')
  plot(resid, main='Residuals y_true - proba_predicted', col=colors.digit[1], pch=16)
  points(which(data_train$y == 1), subset(resid, data_train$y == 1), col=colors.digit[2], pch=16)
  legend('bottomleft', legend=c('digit 5', 'digit 6'), fill=colors.digit)
  interest_indices = which(abs(resid) > 2*sqrt(var(resid)))
  points(interest_indices, resid[interest_indices], col='red', cex=2)
  par(mfrow=c(3, 3))
  for (index in interest_indices) {
    im = matrix(data_train$x[index,], nrow=5, ncol=5, byrow=TRUE)
    im <- t(apply(im, 2, rev))
    image(1:5, 1:5, im, col=gray(0:1), xlab='', ylab='', axes=FALSE, main=sprintf("[%d]: label %d", index, data_train$y[index]))
  }
}

check_residuals()

calc_glm_accuracy <- function() {
  glmod = glm(y ~ x, data=data_train, family='binomial')
  y_predicted = 1 / (1 + exp(-predict.glm(glmod, data_test)))
  y_predicted = as.numeric( y_predicted > 0.5 )
  confusion_matrix = table(data_test$y, y_predicted)
  accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix)
  return(accuracy)
}

glm_accuracy = calc_glm_accuracy()
cat("Non-informative logistic regression test accuracy:", glm_accuracy)

dic.samples(model=mod, n.iter=20)
