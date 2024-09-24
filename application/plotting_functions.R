#if (!require("BiocManager", quietly = TRUE)){ install.packages("BiocManager")}
#BiocManager::install("ggtree")
#BiocManager::install("phyloseq")
#install.packages("remotes")
#remotes::install_github("cpauvert/psadd")

library(fields)
library(ggtree)
library(ggplot2)
library(gplots)
library(gridExtra)
library(Hmsc)
require(phyloseq)
library(psadd)
library(phytools)


reorder_cov <- function(cov_est, cov_qs, tree=NULL, otu_labs=NULL, phylo_reoder=T){
  cov_est_zero = cov_est
  cov_est_zero[(cov_qs[1,,]<0) & (cov_qs[2,,]>0)] <- 0
  colnames(cov_est_zero) <- otu_labs; row.names(cov_est_zero) <- otu_labs
  if(!phylo_reoder){
    return(cov_est_zero)
  }
  tree_labs <- tree$tip.label
  cov_est_zero_r <- cov_est_zero[tree_labs, tree_labs]
  return(cov_est_zero_r)
}

clean_order_taxa <- function(taxonomy){
  taxonomy_plot = taxonomy
  levels(taxonomy_plot$order) <- c(levels(taxonomy_plot$order),"Pseudo Order")
  row.names(taxonomy_plot) <- taxonomy_plot$OTU
  orders <- taxonomy_plot$order
  orders[which(grepl("^pseudo", taxonomy_plot$order, ignore.case = TRUE))] = 'Pseudo Order'
  orders[is.na(orders)] = 'pseudo'
  taxonomy_plot$order_cleaned <- orders
  unique_order <- unique(taxonomy_plot$order_cleaned)
  taxonomy_list <- lapply(unique_order, function(order) {
    taxonomy_plot$OTU[taxonomy_plot$order_cleaned == order]
  })
  names(taxonomy_list) <- unique_order
  return(taxonomy_list)
}

create_regression_coeffs_matrix <- function(Beta_tilde, Vs, rho){
  Beta_est <- Beta_tilde
  Beta_cc_cis <- compute_ci_Beta(Beta_tilde, Vs, rho, alpha=0.05) 
  Beta_est[(Beta_cc_cis$ls<0 & Beta_cc_cis$us>0)] <- 0
  Beta_est[Beta_est>0] = 1
  Beta_est[Beta_est<0] = -1
  return(Beta_est)
}

