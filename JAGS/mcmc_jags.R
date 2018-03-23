library(rjags)
set.seed(26)

read_mnist56 <- function(path_or_url, shuffle = FALSE, n.take = nrow(data)) {
  data = read.table(path_or_url, sep=',', header=TRUE)
  if (shuffle) {
    data = data[sample(nrow(data)),]
  }
  x = data.matrix(data[,-ncol(data)])
  x = scale(x)
  stopifnot(all(!is.na(x)))
  x[x > 0] = 1
  x[x <= 0] = 0
  y = data[, ncol(data)]
  if (!is.null(n.take)) {
    n.take = min(n.take, length(x))
    x = x[1:n.take,]
    y = y[1:n.take]
  }
  data_list = list(x=x, y=y)
  return(data_list)
}

data_train = read_mnist56(url("https://www.dropbox.com/s/l7uppxi1wvfj45z/MNIST56_train.csv?dl=1"), shuffle = TRUE, n.take = 1e3)
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
mod_sim = coda.samples(model=mod, variable.names='w', n.iter=500)
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

plot_convergence <- function(n.step = 10) {
  chain_colors = c('red', 'green', 'blue')
  for (chain_id in 1:3) {
    draws_chain = mod_sim[[chain_id]]
    draws_chain = draws_chain[, startsWith(colnames(draws_chain), 'w')]
    iteration = seq(from=n.step, to=nrow(mod_sim[[1]]), by=n.step)
    epoch_accuracy_func  <- function(epoch) {
      w = as.matrix.2x25(draws_chain[epoch,])
      return(calc_accuracy(w, data_train))
    }
    accuracy = lapply(iteration, epoch_accuracy_func)
    if (chain_id == 1) {
      plot(iteration, accuracy, type='n', ylim=c(0.5, 1), main='Convergence diagnostic')
    }
    points(jitter(iteration, amount = 0.5), accuracy, col=chain_colors[chain_id])
  }
  legend('bottomright', legend=c('chain 1', 'chain 2', 'chain 3'), col=chain_colors, fill=chain_colors)
}
plot_convergence(n.step=2)

check_residuals <- function() {
  z = data_train$x %*% t(w)
  py = 1 / (1 + exp(z[,1] -z[,2]))
  resid = data_train$y - py
  plot(jitter(resid, amount=0.01), main='Residuals y_true - proba_predicted')
  plot(jitter(py, amount=0.02), jitter(resid, amount=0.02))
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
