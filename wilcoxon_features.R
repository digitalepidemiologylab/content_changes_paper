library(stats)

significance_level <- 0.05
collective_results_folder <- file.path( 
                      "results_statistical_analysis", "run",
                      "collective_tests")
prepost_filepaths <- list.files(path=collective_results_folder, pattern="\\.csv$")
# There should be 61 CSV files: 
# - 1 CSV file for the pre/post comparison on the weekly rates of tweets posted (full volume)
# - 60 CSV files for the different features: 
#       - Emotions: 12 files
#       - Topics: 10 files
#       - URL categories: 26 files (including Undefined)
#       - Symptoms: 12 files

names_columns <- c("feature", "pseudomedian", "lb_ci", "ub_ci", "pvalue", "pre_post")
stats_results_df <- data.frame(matrix(ncol = length(names_columns), nrow = 0, 
                    dimnames = list(NULL, names_columns)))


for (fpath in prepost_filepaths){
  feature <- gsub("\\.csv$", "", basename(fpath))
  feature <- gsub("_pre_post_fractions$", "", feature)
  feature <- gsub("_pre_post_rates$", "", feature)
  print(feature)
  pre_post_df <- read.csv(file.path(collective_results_folder, fpath))

  # Apply two-sided Wilcoxon signed-rank test
  res <- wilcox.test(pre_post_df$avg_post, pre_post_df$avg_pre, paired=TRUE, conf.int=TRUE, conf.level=0.95, exact=FALSE)
  if (res$p.value < significance_level){
    if (res$estimate < 0) {
      change_test <- "decrease"
    } else {
      change_test <- "increase"
    }
  } else {
    change_test <- "no_change"
  }
  stats_results_df[nrow(stats_results_df)+1,] <- c(feature, res$estimate, 
                                                  res$conf.int[1], res$conf.int[2], 
                                                  res$p.value, change_test)
} 

write.xlsx(stats_results_df, file = file.path(collective_results_folder, "wilcoxon_results_all_weeks_before_adjustment.xlsx"), 
          row.names=FALSE)
