library(ggplot2)
library(reshape2)
library(ggdendro)
library(grid)
library(gridExtra)
library(readr)
library(viridis)
library(cowplot)

base_Dir <- 'F:/'
talker_List <- c("Agnes", "Alex", "Bruce", "Fred", "Junior", "Kathy", "Princess", "Ralph", "Vicki", "Victoria")
epoch_List <- c(600)
hidden_Type <- 'LSTM'
hidden_Unit <- 512
index <- 0

for (epoch in epoch_List)
{
  for (talker in talker_List)
  {
    work_Dir <- sprintf('%sHT_%s.HU_%s.ET_%s.IDX_%s/Hidden_Analysis/E.%s/', base_Dir, hidden_Type, hidden_Unit, talker, index, epoch)
    
    start_Window <- 0
    end_Window <- 35
    for (flow_Type in c("Phoneme", "Feature"))
    {
      if (!dir.exists(paste(work_Dir, "Flow.", flow_Type, "/PNG", sep="")))
      {
        dir.create(paste(work_Dir, "Flow.", flow_Type, "/PNG", sep=""))
      }
      
      plot_List <- list()
      for (unit_Index in seq(0, hidden_Unit - 1, 1))
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
          labs(title=sprintf('Phoneme flow    Unit: %s', unit_Index), x= 'Time (ms)', y= flow_Type, fill="") +
          theme(
            title = element_text(size=20),
            axis.title.x = element_text(size=20),
            axis.title.y = element_text(size=20),
            axis.title.y.right = element_text(size=20),
            axis.text.x = element_text(size=18),
            axis.text.y = element_text(size=18),
            axis.ticks = element_blank(),
            legend.position="right",
            legend.direction="vertical",
            legend.key.height = unit(20, "mm"),
            plot.margin=unit(c(0,0,0,0),"cm"),
            panel.grid=element_blank()
          )
        
        if (flow_Type == "Phoneme")
        {
          ggsave(
            filename = paste(work_Dir, "Flow.", flow_Type, "/PNG/W_(", start_Window,",", end_Window,").", flow_Type, ".U_", unit_Index, ".T_All.png", sep=""),
            plot = plot,
            device = "png",
            width = 15,
            height = 25,
            units = "cm",
            dpi = 300
          )
        }
        if (flow_Type == "Feature")
        {
          ggsave(
            filename = paste(work_Dir, "Flow.", flow_Type, "/PNG/W_(", start_Window,",", end_Window,").", flow_Type, ".U_", unit_Index, ".T_All.png", sep=""),
            plot = plot,
            device = "png",
            width = 20,
            height = 25, #10,
            units = "cm",
            dpi = 300
          )
        }
      }
    }
  }
}