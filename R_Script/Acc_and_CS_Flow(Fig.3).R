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

flow_Plot <- function(flow_Data, epoch)
{
  flow_Data$Pattern_Type <- factor(
    flow_Data$Pattern_Type,
    levels= c('Trained', 'Pattern_Excluded', 'Talker_Excluded'),
    labels= c('Trained', 'Excluded pattern', 'Excluded talker')
    )
  flow_Data$Category <- factor(
    flow_Data$Category,
    levels= c('Target', 'Cohort', 'Rhyme', 'Unrelated'),
    labels= c('Target', 'Cohort', 'Rhyme', 'Unrelated')
    )
  
  plot <- ggplot(data= flow_Data, aes(x=Time_Step, y=Cosine_Similarity, color=Category, shape=Category)) +
    geom_line() +
    geom_point(data=subset(flow_Data, Time_Step %% 10 == 0), aes(x=Time_Step, y=Cosine_Similarity, color=Category, shape=Category), size = 3) +
    scale_x_continuous(breaks = seq(0, max_Display_Step, max_Display_Step / 4), labels = seq(0, max_Display_Step, max_Display_Step / 4)*10) +
    facet_grid(.~Pattern_Type) +
    ylim(0,1.1) +
    labs(title= sprintf('CS flow    Epoch: %s', epoch), x='Time step', y='Cosine similarity') +
    theme_bw() +
    theme(
      text = element_text(size=16),
      panel.background = element_blank(),
      panel.grid.major = element_blank(),  #remove major-grid labels
      panel.grid.minor = element_blank(),  #remove minor-grid labels
      plot.background = element_blank(),
      plot.title = element_text(hjust = 0.5)
      )
  
  return(plot)
}

flow_Plot2 <- function(flow_Data, epoch)#No distinguish pattern type
{
  flow_Data$Category <- factor(
    flow_Data$Category,
    levels= c('Target', 'Cohort', 'Rhyme', 'Unrelated'),
    labels= c('Target', 'Cohort', 'Rhyme', 'Unrelated')
  )
  
  plot <- ggplot(data= flow_Data, aes(x=Time_Step, y=Cosine_Similarity, color=Category, shape=Category)) +
    geom_line() +
    geom_point(data=subset(flow_Data, Time_Step %% 10 == 0), aes(x=Time_Step, y=Cosine_Similarity, color=Category, shape=Category), size = 3) +
    scale_x_continuous(breaks = seq(0, max_Display_Step, max_Display_Step / 4), labels = seq(0, max_Display_Step, max_Display_Step / 4)*10) +
    ylim(0,1.1) +
    labs(title= sprintf('CS flow    Epoch: %s', epoch), x='Time step', y='Cosine similarity', colour='', shape='') +
    theme_bw() +
    theme(
      text = element_text(size=12),
      panel.background = element_blank(),
      panel.grid.major = element_blank(),  #remove major-grid labels
      panel.grid.minor = element_blank(),  #remove minor-grid labels
      plot.background = element_blank(),
      legend.position = 'bottom',
      legend.text = element_text(size=7),
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
talker_List <- c('Agnes', 'Alex', 'Bruce', 'Fred', 'Junior', 'Kathy', 'Princess', 'Ralph', 'Vicki', 'Victoria')
epoch_with_Exclusion <- 600
epoch_without_Exclusion <- 800
max_Display_Step <- 60
hidden_Type <- 'LSTM'
hidden_Unit <- 512
index <- 0


acc_List <- list()
flow_with_Exclusion_List <- list()
flow_without_Exclusion_List <- list()

for (talker in talker_List)
{
  work_Dir <- sprintf('%sHT_%s.HU_%s.ET_%s.IDX_%s/Result/', base_Dir, hidden_Type, hidden_Unit, talker, index)
  rt_Data <- read_delim(paste(work_Dir, 'RT_Result.txt', sep=""), "\t", escape_double = FALSE, locale = locale(encoding = "UTF-8"), trim_ws = TRUE)
  
  rt_Data <- rt_Data[c(1,4,12)]
  rt_Data$Accuracy <- as.numeric(!is.nan(rt_Data$Onset_Time_Dependent_RT))
  rt_Data$Excluded_Talker <- talker
  
  acc_List[[length(acc_List) + 1]] <- rt_Data
  
  for(epoch in c(epoch_with_Exclusion, epoch_without_Exclusion))
  {
    categorized_Flow_Data <- read_delim(paste(work_Dir, sprintf('Categorized_Flow/Categorized_Flow.E_%s.txt', epoch), sep=""), "\t", escape_double = FALSE, locale = locale(encoding = "UTF-8"), trim_ws = TRUE)
    categorized_Flow_Data <- subset(na.omit(categorized_Flow_Data), Accuracy)[-c(1,2,3,5,6,8,9)]
    if (nrow(categorized_Flow_Data) == 0)
    {
      next
    }

    categorized_Flow_Data.Mean <- aggregate(categorized_Flow_Data[3:length(categorized_Flow_Data)], by=list(categorized_Flow_Data$Pattern_Type, categorized_Flow_Data$Category), FUN=mean)
    colnames(categorized_Flow_Data.Mean)[1:2] <- c('Pattern_Type', 'Category')
    categorized_Flow_Data.Melt <- melt(categorized_Flow_Data.Mean, id.vars = c('Pattern_Type', 'Category'), variable.name = 'Time_Step', value.name = 'Cosine_Similarity')
    categorized_Flow_Data.Melt$Time_Step <- as.numeric(categorized_Flow_Data.Melt$Time_Step)
    categorized_Flow_Data.Melt <- subset(categorized_Flow_Data.Melt, Time_Step <= max_Display_Step)
    categorized_Flow_Data.Melt$Excluded_Talker <- talker

    if (epoch == epoch_with_Exclusion)
    {
      flow_with_Exclusion_List[[length(flow_with_Exclusion_List) + 1]] <- categorized_Flow_Data.Melt
    }
    if (epoch == epoch_without_Exclusion)
    {
      flow_without_Exclusion_List[[length(flow_without_Exclusion_List) + 1]] <- categorized_Flow_Data.Melt
    }
  }
}

acc_Data <- do.call(rbind, acc_List)
flow_with_Exclusion_Data <- do.call(rbind, flow_with_Exclusion_List)
flow_without_Exclusion_Data <- do.call(rbind, flow_without_Exclusion_List)

acc_Plot_List <- list()
flow_with_Exclusion_Plot_List <- list()
flow_without_Exclusion_Plot_List <- list()

for (talker in talker_List)
{
  work_Dir <- sprintf('%sHT_%s.HU_%s.ET_%s.IDX_%s/Result/', base_Dir, hidden_Type, hidden_Unit, talker, index)
  
  acc_Data.Subset <- subset(acc_Data, Excluded_Talker == talker)
  plot <- acc_Plot(acc_Data.Subset, talker)
  ggsave(filename = paste(work_Dir, "Accuracy_Flow.png", sep=""), plot = plot, device = "png", width = 12, height = 12, units = "cm", dpi = 300)
  acc_Plot_List[[length(acc_Plot_List) + 1]] <- plot
  
  flow_with_Exclusion_Data.Subset <- subset(flow_with_Exclusion_Data, Excluded_Talker == talker)
  plot <- flow_Plot(flow_with_Exclusion_Data.Subset, epoch_with_Exclusion)
  ggsave(filename = paste(work_Dir, sprintf('Categorized_Flow.E_%s.png', epoch_with_Exclusion), sep=""), plot = plot, device = "png", width = 38, height = 12, units = "cm", dpi = 300)
  flow_with_Exclusion_Plot_List[[length(flow_with_Exclusion_Plot_List) + 1]] <- plot

  flow_without_Exclusion_Data.Subset <- subset(flow_without_Exclusion_Data, Excluded_Talker == talker)
  plot <- flow_Plot(flow_without_Exclusion_Data.Subset, epoch_without_Exclusion)
  ggsave(filename = paste(work_Dir, sprintf('Categorized_Flow.E_%s.png', epoch_without_Exclusion), sep=""), plot = plot, device = "png", width = 38, height = 12, units = "cm", dpi = 300)
  flow_without_Exclusion_Plot_List[[length(flow_without_Exclusion_Plot_List) + 1]] <- plot
}

png(paste(base_Dir, sprintf('Accuracy_Flow.IDX_%s.All.png', index), sep=""), width = 40, height = 22, res =300, units = "cm")
grid.arrange(arrangeGrob(grobs = acc_Plot_List, ncol=5))
dev.off()

png(paste(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.All.png', index, epoch_with_Exclusion), sep=""), width = 120, height = 40, res =300, units = "cm")
grid.arrange(arrangeGrob(grobs =flow_with_Exclusion_Plot_List, ncol=5))
dev.off()

png(paste(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.All.png', index, epoch_without_Exclusion), sep=""), width = 120, height = 20, res =300, units = "cm")
grid.arrange(arrangeGrob(grobs =flow_without_Exclusion_Plot_List, ncol=5))
dev.off()

ggsave(
  filename = paste(base_Dir, sprintf('Accuracy_Flow.IDX_%s.Avg.png', index), sep=""),
  plot = acc_Plot(acc_Data, 'All'),
  device = "png", width = 8, height = 12, units = "cm", dpi = 300
  )
ggsave(
  filename = paste(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.Avg.png', index, epoch_with_Exclusion), sep=""),
  plot = flow_Plot(data_summary(flow_with_Exclusion_Data, 'Cosine_Similarity', c('Pattern_Type', 'Category', 'Time_Step')), epoch_with_Exclusion),
  device = "png", width = 26, height = 12, units = "cm", dpi = 300
  )
ggsave(
  filename = paste(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.Avg.png', index, epoch_without_Exclusion), sep=""),
  plot = flow_Plot(data_summary(flow_without_Exclusion_Data, 'Cosine_Similarity', c('Pattern_Type', 'Category', 'Time_Step')), epoch_without_Exclusion),
  device = "png", width = 26, height = 12, units = "cm", dpi = 300
  )

ggsave(
  filename = paste(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.Avg.No_Pattern_Type.png', index, epoch_with_Exclusion), sep=""),
  plot = flow_Plot2(data_summary(flow_with_Exclusion_Data, 'Cosine_Similarity', c('Category', 'Time_Step')), epoch_with_Exclusion),
  device = "png", width = 8, height = 12, units = "cm", dpi = 300
)
ggsave(
  filename = paste(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.Avg.No_Pattern_Type.png', index, epoch_without_Exclusion), sep=""),
  plot = flow_Plot2(data_summary(flow_without_Exclusion_Data, 'Cosine_Similarity', c('Category', 'Time_Step')), epoch_without_Exclusion),
  device = "png", width = 8, height = 12, units = "cm", dpi = 300
)

