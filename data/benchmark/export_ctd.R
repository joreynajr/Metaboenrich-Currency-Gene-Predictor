
args <- commandArgs(trailingOnly = TRUE)
pkg <- args[1]; out <- args[2]
env <- new.env()
for (f in c("Thistlethwaite2020", "Miller2015", "cohorts_coded")) {
  load(file.path(pkg, "data", paste0(f, ".RData")), envir = env)
}
write.csv(env$Thistlethwaite2020, file.path(out, "thistlethwaite2020.csv"))
write.csv(env$Miller2015, file.path(out, "miller2015.csv"))
cc <- env$cohorts_coded
long <- data.frame(cohort = rep(names(cc), lengths(cc)), sample = unlist(cc, use.names = FALSE))
write.table(long, file.path(out, "cohorts_coded.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
