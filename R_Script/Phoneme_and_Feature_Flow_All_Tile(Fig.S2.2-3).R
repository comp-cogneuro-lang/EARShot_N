Sort_Reference_by_PSI <- function(path){
  library(readr)
  library(ggdendro)
  
  map_Data <- read_delim(path, "\t", escape_double = FALSE, locale = locale(encoding = "UTF-8"), trim_ws = TRUE)[,-1]
  rownames(map_Data) <- rownames(read.table(file=path, row.names = 1, header = TRUE, encoding="UTF-8"))
  
  x <- as.matrix(scale(map_Data))
  x[x=="NaN"] = 0
  sorted_Unit_Index_List <- order.dendrogram(as.dendrogram(hclust(dist(t(x))))) - 1
  
  return(sorted_Unit_Index_List)
}

library(ggplot2)
library(reshape2)
library(grid)
library(gridExtra)
library(readr)
library(viridis)

base_Dir <- 'F:/'
talker_List <- c("Agnes", "Alex", "Bruce", "Fred", "Junior", "Kathy", "Princess", "Ralph", "Vicki", "Victoria")
epoch_List <- c(600)
hidden_Type <- 'LSTM'
hidden_Unit <- 512
index <- 0
reference_PSI_Epoch <- 600
reference_PSI_Criterion <- 0.0

for (epoch in epoch_List)
{
  for (talker in talker_List)
  {
    sorted_Unit_Index_List <- Sort_Reference_by_PSI(sprintf(
      '%sHT_%s.HU_%s.ET_%s.IDX_%s/Hidden_Analysis/E.%s/Map.PSI/TXT/W_(5,15).Normal.PSI.C_%s.D_Positive.T_All.txt',
      base_Dir, hidden_Type, hidden_Unit, talker, index, reference_PSI_Epoch, format(reference_PSI_Criterion, nsmall=2)
      ))
    
    work_Dir <- sprintf('%sHT_%s.HU_%s.ET_%s.IDX_%s/Hidden_Analysis/E.%s/', base_Dir, hidden_Type, hidden_Unit, talker, index, epoch)
    start_Window <- 0
    end_Window <- 35
    for (flow_Type in c("Phoneme", "Feature"))
    {
      plot_List <- list()
      for (unit_Index in sorted_Unit_Index_List)
      {
        flow_Data <- read_delim(paste(work_Dir, "Flow.", flow_Type, "/TXT/W_(", start_Window,",", end_Window,").", flow_Type, ".U_", unit_Index, ".T_All.txt", sep=""), "\t", escape_double = FALSE, locale = locale(encoding = "UTF-8"), trim_ws = TRUE)[,-1]
        rownames(flow_Data) <- rownames(read.table(file=paste(work_Dir, "Flow.", flow_Type, "/TXT/W_(", start_Window,",", end_Window,").", flow_Type, ".U_", unit_Index, ".T_All.txt", sep=""), row.names = 1, header = TRUE, encoding="UTF-8"))
        
        mean_Flow_Data <- colMeans(flow_Data)
        mean_Flow_Data <- as.data.frame(mean_Flow_Data)
        colnames(mean_Flow_Data) <- c("Mean")
        mean_Flow_Data$Step <- as.numeric(rownames(mean_Flow_Data))
        
        col_Min <- min(as.numeric(colnames(flow_Data)), na.rm = TRUE)
        col_Max <- max(as.numeric(colnames(flow_Data)), na.rm = TRUE)
        
        
        flow_Data$row_Name.num <- rev(1:length(rownames(flow_Data)))
        key.flow_Data.row_Name <- data.frame(row_Name = rownames(flow_Data), row_Name.num = (1:length(rownames(flow_Data))))
        mdf <- melt(as.data.frame(flow_Data), id.vars="row_Name.num")
        mdf <- merge(mdf, key.flow_Data.row_Name, by = "row_Name.num", all.x = TRUE)
        ylabels = rev(rownames(flow_Data))
        
        plot <- ggplot(mdf, aes(x=variable, y=row_Name.num)) +
          geom_tile(aes(fill=value)) +
          scale_fill_viridis(option="plasma", limits=c(0, 1), breaks=c(0, 1),labels=c(0, 1)) +
          scale_x_discrete(
            breaks = c(col_Min, seq(0, col_Max, by = 5), col_Max),
            labels = c(col_Min, seq(0, col_Max, by = 5), col_Max) * 10
          ) +
          scale_y_continuous(
            expand=c(0,0),
            breaks = seq(1, max(mdf$row_Name.num), by = 1),
            labels = ylabels,
            sec.axis = dup_axis()
          ) +
          labs(title='', x= '', y= '', fill='') +
          theme(
            title = element_blank(),
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            axis.ticks = element_blank(),
            legend.position="none",
            panel.grid=element_blank()
            )
        
        plot_List[[length(plot_List) + 1]] <- plot
      }
      
      if (!dir.exists(paste(work_Dir, "Flow.", flow_Type, "/PNG.Tile", sep="")))
      {
        dir.create(paste(work_Dir, "Flow.", flow_Type, "/PNG.Tile", sep=""))
      }
      
      margin = theme(plot.margin = unit(c(-0.02,-0.05,-0.02,-0.05), "cm"))
      ggsave(
        filename = sprintf('%sFlow.%s/PNG.Tile/%s.Flow_Tile.png', work_Dir, flow_Type, flow_Type),
        plot = grid.arrange(grobs = lapply(plot_List[1:length(sorted_Unit_Index_List)], "+", margin), ncol=21),
        device = "png",
        width = 21.6,
        height = 28,
        units = "cm",
        dpi = 300
      )
    }
  }
}