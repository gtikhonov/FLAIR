
source('plotting_functions.R')

## -REGRESSION COEFFICENTS PLOTS ####

# taxonomical tree plot
prev = colSums(Y)
sel.sp = prev>=th
taxonomy.sel = taxonomy[sel.sp,]
taxonomy.sel$OTU = as.factor(taxonomy.sel$OTU)
phy.tree = as.phylo.formula(
  ~kingdom/phylum/order/order/family/subfamily/tribe/genus/OTU, data=taxonomy.sel
)
ggtree(phy.tree.2, cex=0.03)
order_taxa_list <- clean_order_taxa(taxonomy.sel)

groupOTU(plot_tree, order_taxa_list, 'Order') +
  aes(color = Order) +
  scale_color_discrete(drop = TRUE, 
                       limits = function(x) x[x != "0"])  +
  theme(legend.position = "left")  


Beta_est_insects <- create_regression_coeffs_matrix(
  flair.insects.th15.full$Beta_tilde, 
  flair.insects.th15.full$Vs[1:ncol(X)], 
  flair.insects.th15.full$rho_max
  )
image.plot(t(Beta_est_insects[,-c(1,2)]), axes=F, legend.width = 0,legend.shrink=0,
           col = colorRampPalette(c("blue","white","red"))(200))


## - COVARIANCE PLOT ####

### - random 1000 + species names ####

flair.insects.th15.full.Lambda_outer.random <- sample_Lambda_outer(flair.insects.th15.full.samples.random$Lambda_samples,flair.insects.th15.full.samples.random$Lambda_hat, rho=1)
flair_insects_Lambda_outer_qs_random_1000 <- apply(flair.insects.th15.full.Lambda_outer.random$Lambda_outer_samples_cc, c(1,2), function(x)(quantile(x, probs=c(alpha/2, 1-alpha/2))))
phy_tree_random = as.phylo.formula(~kingdom/phylum/order/order/family/subfamily/tribe/genus/OTU,data=taxonomy.sel[subsample_index,])
otu_labs_random <- taxonomy.sel[subsample_index,]$OTU
flair_insects_covs_random <- reorder_cov(flair.insects.th15.full$Lambda_outer_mean_cc[subsample_index, subsample_index], 
                                         flair_insects_Lambda_outer_qs_random_1000, phy_tree_random, 
                                         otu_labs_random)
dev.new()
gplots::heatmap.2(
  flair_insects_covs_random, scale = "none", col = bluered(100), trace = "none",
  density.info = "none", Rowv=F, Colv=F, symm=T, revC=T
  )
dev.new()
gplots::heatmap.2(
  flair_insects_covs_random, scale = "none", col = bluered(100), trace = "none", density.info = "none"
  )



### - W/ SPECIES NAMES ####
subsample_index_w_names <- subsample_index[-c(1:1000)][1:50]
labs_df_species  <- data.frame(
  'species_names' = taxonomy.sel[subsample_index, 'species'],
  'otus' = taxonomy.sel[subsample_index, 'OTU']
  )
labs_df_species[-c(1:1000),]
flair_insects_covs_random_w_names <- reorder_cov(
  flair.insects.th15.full$Lambda_outer_mean_cc[subsample_index_w_names, subsample_index_w_names], 
  flair_insects_Lambda_outer_qs_random_1000[,-c(1:1000), -c(1:1000)][, 1:50, 1:50], 
  tree=NULL, labs_df_species[-c(1:1000),]$species_names[1:50], phylo_reoder=F)
dev.new()
gplots::heatmap.2(flair_insects_covs_random_w_names, scale = "none", col = bluered(100), 
                  trace = "none", density.info = "none")




