data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE),
      se = sd(x[[col]], na.rm=TRUE) / sqrt(length(x[[col]])))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}

acc_Plot <- function(acc_Data, talker)
{
  summary <- data_summary(acc_Data, 'Accuracy', c('Epoch', 'Pattern_Type'))
  summary$Pattern_Type <- factor(
    summary$Pattern_Type,
    levels= c('Trained', 'Pattern_Excluded', 'Talker_Excluded'),
    labels= c('Trained', 'Excluded pattern', 'Excluded talker')
  )
  
  plot <- ggplot(data= summary, aes(x=Epoch, y=Accuracy, color=Pattern_Type, shape=Pattern_Type)) +
    geom_point(size=3) +
    geom_line(data= subset(summary, Epoch <= epoch_with_Exclusion), aes(x=Epoch, y=Accuracy, color=Pattern_Type, shape=Pattern_Type)) +
    geom_line(data= subset(summary, Epoch > epoch_with_Exclusion), aes(x=Epoch, y=Accuracy, color=Pattern_Type, shape=Pattern_Type)) +
    #geom_text(data= summary, aes(x=Epoch, y=Accuracy + 0.05, label=round(Accuracy, 3))) +
    geom_errorbar(aes(ymin=Accuracy-se, ymax=Accuracy+se), width= max(summary$Epoch) * 0.15, position=position_dodge(0.05)) +
    labs(title= talker, x="Epoch", y= "Accuracy", colour='Pattern type', shape='Pattern type') +
    ylim(0, 1.1) +
    theme_bw() +
    theme(
      axis.title.x = element_text(size=20),
      axis.title.y = element_text(size=20),
      axis.text.x = element_text(size=13),
      axis.text.y = element_text(size=13),
      panel.grid=element_blank(),
      legend.title = element_text(size=1),
      legend.text = element_text(size=6),
      legend.position = 'bottom',
      plot.title = element_text(hjust = 0.5)
    )
  
  return(plot)
}

library(readr)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(grid)

base_Dir <- 'F:/'
talker_List <- c('Agnes')#, 'Alex', 'Bruce', 'Fred', 'Junior', 'Kathy', 'Princess', 'Ralph', 'Vicki', 'Victoria')
epoch_with_Exclusion <- 600
epoch_without_Exclusion <- 800
max_Display_Step <- 60
hidden_Type <- 'LSTM'
hidden_Unit <- 512
index <- 0

acc_List <- list()
for (talker in talker_List)
{
  work_Dir <- sprintf('%sHT_%s.HU_%s.ET_%s.IDX_%s/Result/', base_Dir, hidden_Type, hidden_Unit, talker, index)
  rt_Data <- read_delim(paste(work_Dir, 'RT_Result.txt', sep=""), "\t", escape_double = FALSE, locale = locale(encoding = "UTF-8"), trim_ws = TRUE)
  
  rt_Data <- rt_Data[c(1,3,4,12)]
  rt_Data$Accuracy <- as.numeric(!is.nan(rt_Data$Onset_Time_Dependent_RT))
  
  acc_List[[length(acc_List) + 1]] <- rt_Data
}

acc_Data <- do.call(rbind, acc_List)

acc_Plot_List <- list()
for (talker in talker_List)
{
  #acc_Data.Subset <- subset(acc_Data, acc_Data$Talker == talker)
  acc_Data.Subset <- subset(acc_Data, toupper(acc_Data$Talker) == toupper(talker))
  plot <- acc_Plot(acc_Data.Subset, talker)
  acc_Plot_List[[length(acc_Plot_List) + 1]] <- plot
}

png(paste(base_Dir, sprintf('Accuracy_Flow.IDX_%s.Integration_by_Talker.png', index), sep=""), width = 40, height = 22, res =300, units = "cm")
grid.arrange(arrangeGrob(grobs = acc_Plot_List, ncol=5))
dev.off()