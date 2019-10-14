#Dendrogram refer:
#https://stackoverflow.com/questions/6673162/reproducing-lattice-dendrogram-graph-with-ggplot2
#https://stackoverflow.com/questions/48664746/how-to-set-two-x-axis-and-two-y-axis-using-ggplot2


library(ggplot2)
library(reshape2)
library(ggdendro)
library(grid)
library(gridExtra)
library(readr)
library(viridis)

get_legend<-function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}


### Set up a blank theme
theme_none <- theme(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  panel.background = element_blank(),
  axis.line = element_blank(),
  #axis.ticks.length = element_blank()
  plot.margin=unit(c(0,0,0,0),"cm"),
  panel.grid=element_blank(),
  axis.title=element_blank(),
  axis.ticks=element_blank(),
  axis.text=element_blank()
)

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
    
  
    for (map_Type in c("PSI", "FSI"))
    {
      if (!dir.exists(paste(work_Dir, "Map.", map_Type, "/PNG/", sep="")))
      {
        dir.create(paste(work_Dir, "Map.", map_Type, "/PNG/", sep=""))
      }
      if (!dir.exists(paste(work_Dir, "Map.", map_Type, "/PNG.NoColSort/", sep="")))
      {
        dir.create(paste(work_Dir, "Map.", map_Type, "/PNG.NoColSort/", sep=""))
      }
    }
    
    for (window_Range in c(10))
    {
      for (start_Window in c(5))
      {
        end_Window = start_Window + window_Range
        for (criterion in round(seq(0, 0.50, 0.01), 2))
        {
          criterion <- format(round(criterion, 2), nsmall = 2)
          
          for (map_Type in c("PSI", "FSI"))
          {
            if (map_Type == "PSI") { row_Title="Phoneme" }
            if (map_Type == "FSI") { row_Title="Feature" }
            
            for (direction in c("Positive"))
            {
              map_Data <- read_delim(paste(work_Dir, "Map.", map_Type, "/TXT/W_(", start_Window,",", end_Window,").Squeezed.", map_Type, ".C_", criterion, ".D_", direction, ".T_All.txt", sep=""), "\t", escape_double = FALSE, locale = locale(encoding = "UTF-8"), trim_ws = TRUE)[,-1]
              rownames(map_Data) <- rownames(read.table(file=paste(work_Dir, "Map.", map_Type, "/TXT/W_(", start_Window,",", end_Window,").Squeezed.", map_Type, ".C_", criterion, ".D_", direction, ".T_All.txt", sep=""), row.names = 1, header = TRUE, encoding="UTF-8"))
              
              map_Limit <- nrow(map_Data) - 1
              
              if (sum(map_Data) == 0)
              {
                next
              }
              
              x <- as.matrix(scale(map_Data))
              
              if (ncol(x) < 2) { next }
              
              dd.row <- as.dendrogram(hclust(dist(t(x))))
              row.ord <- order.dendrogram(dd.row)
              
              xx <- scale(map_Data)[, row.ord]
              xx_names <- attr(xx, "dimnames")
              xx <- map_Data[, row.ord]
              rownames(xx) <- rownames(scale(map_Data)[, row.ord])
              df <- as.data.frame(xx)
              colnames(df) <- xx_names[[2]]
              df$row_Name = xx_names[[1]]
              df$row_Name <- with(df, factor(row_Name, levels=row_Name, ordered=TRUE))
              
              mdf <- melt(df, id.vars="row_Name")
              
              key.mdf.row_Name <- data.frame(row_Name = rownames(map_Data), row_Name.num = (1:length(rownames(map_Data))))
              mdf <- merge(mdf, key.mdf.row_Name, by = "row_Name", all.x = TRUE)
              ylabels = rownames(map_Data)
              
              ddata_x <- dendro_data(dd.row)
              
              
              p1 <- ggplot(mdf, aes(x=variable, y=row_Name.num)) +
                geom_tile(aes(fill=value)) +
                scale_fill_viridis(option="plasma", limits=c(0, map_Limit), breaks=c(0, map_Limit),labels=c(0, map_Limit)) +
                scale_y_continuous(
                  trans = "reverse",
                  expand=c(0,0),
                  breaks = seq(1, max(mdf$row_Name.num), by = 1),
                  labels = ylabels,
                  sec.axis = dup_axis()
                ) +
                labs(y=row_Title, fill="") +
                theme(
                  axis.title.x = element_blank(),
                  axis.title.y = element_text(size=20),
                  axis.title.y.right = element_blank(),
                  axis.text.x = element_blank(),
                  axis.text.y = element_text(size=15),
                  axis.ticks = element_blank(),
                  legend.position=c(1.07, -0.15),
                  legend.direction="horizontal",
                  legend.text = element_text(size=20),
                  plot.margin=unit(c(0,0,0,0),"cm"),
                  panel.grid=element_blank()
                )
              
              # Dendrogram 1
              p2 <- ggplot(segment(ddata_x)) +
                geom_segment(aes(x=x, y=y, xend=xend, yend=yend)) +
                theme_none + theme(axis.title.x=element_blank()) + scale_y_reverse()
              
              if (map_Type == "PSI")
              {
                
                png(paste(work_Dir, 'Map.PSI/PNG.NoColSort/E_', epoch,".PSI_No_Column_Sort.W_(", start_Window,",", end_Window,").T_All.D_", direction,".C_", criterion, ".in_R.png", sep=""), width = 270 * 1.2, height = 250, res =300, units = "mm")
                grid.newpage()
                print(p2, vp=viewport(0.915, 0.2, x=0.46, y=0.095))
                print(p1, vp=viewport(0.9, 0.8, x=0.45, y=0.59))
                dev.off()
              }
              if (map_Type == "FSI")
              {
                png(paste(work_Dir, 'Map.FSI/PNG.NoColSort/E_', epoch,".FSI_No_Column_Sort.W_(", start_Window,",", end_Window,").T_All.D_", direction,".C_", criterion, ".in_R.png", sep=""), width = 350 * 1.2, height = 130, res =300, units = "mm")
                grid.newpage()
                print(p2, vp=viewport(0.83, 0.2, x=0.457, y=0.095))
                print(p1, vp=viewport(0.9, 0.8, x=0.45, y=0.59))
                dev.off()
              }
              
              
              x <- as.matrix(scale(map_Data))
              dd.col <- as.dendrogram(hclust(dist(x)))
              col.ord <- order.dendrogram(dd.col)
              
              dd.row <- as.dendrogram(hclust(dist(t(x))))
              row.ord <- order.dendrogram(dd.row)
              
              xx <- scale(map_Data)[col.ord, row.ord]
              xx_names <- attr(xx, "dimnames")
              xx <- map_Data[col.ord, row.ord]
              rownames(xx) <- rownames(scale(map_Data)[col.ord, row.ord])
              df <- as.data.frame(xx)
              colnames(df) <- xx_names[[2]]
              df$row_Name = xx_names[[1]]
              df$row_Name <- with(df, factor(row_Name, levels=row_Name, ordered=TRUE))
              
              mdf <- melt(df, id.vars="row_Name")
              
              key.mdf.row_Name <- data.frame(row_Name = rownames(map_Data)[col.ord], row_Name.num = (1:length(rownames(map_Data))))
              mdf <- merge(mdf, key.mdf.row_Name, by = "row_Name", all.x = TRUE)
              ylabels = rownames(map_Data)[col.ord]
              
              ddata_x <- dendro_data(dd.row)
              ddata_y <- dendro_data(dd.col)
              
              
              
              ### Create plot components ###
              # Heatmap
              p1 <- ggplot(mdf, aes(x=variable, y=row_Name.num)) +
                geom_tile(aes(fill=value)) +
                scale_fill_viridis(option="plasma", limits=c(0, map_Limit), breaks=c(0, map_Limit),labels=c(0, map_Limit)) +
                scale_y_continuous(
                  expand=c(0,0),
                  breaks = seq(1, max(mdf$row_Name.num), by = 1),
                  labels = ylabels,
                  sec.axis = dup_axis()
                ) +
                labs(y=row_Title, fill="") +
                theme(
                  axis.title.x = element_blank(),
                  axis.title.y = element_text(size=20),
                  axis.title.y.right = element_blank(),
                  axis.text.x = element_blank(),
                  axis.text.y = element_text(size=15),
                  axis.ticks = element_blank(),
                  legend.position=c(1.07, -0.15),
                  legend.direction="horizontal",
                  legend.text = element_text(size=20),
                  plot.margin=unit(c(0,0,0,0),"cm"),
                  panel.grid=element_blank()
                )
              
              
              
              # Dendrogram 1
              p2 <- ggplot(segment(ddata_x)) +
                geom_segment(aes(x=x, y=y, xend=xend, yend=yend)) +
                theme_none + theme(axis.title.x=element_blank()) + scale_y_reverse()
              
              # Dendrogram 2
              p3 <- ggplot(segment(ddata_y)) +
                geom_segment(aes(x=x, y=y, xend=xend, yend=yend)) +
                coord_flip() +
                theme_none +
                theme(axis.title.x=element_blank())
              geom_text(data=label(ddata_y), aes(label=label, x=x, y=0), hjust=0.5,size=3)
              
              ### Draw graphic ###
              if (map_Type == "PSI")
              { 
                png(paste(work_Dir, "Map.PSI/PNG/E_", epoch,".PSI_with_Dendrogram.W_(", start_Window,",", end_Window,").T_All.D_", direction,".C_", criterion, ".in_R.png", sep=""), width = 270 * 1.2, height = 250, res =300, units = "mm")
                grid.newpage()
                print(p2, vp=viewport(0.915, 0.2, x=0.46, y=0.095))
                print(p3, vp=viewport(0.1, 0.855, x=0.95, y=0.59))
                print(p1, vp=viewport(0.9, 0.8, x=0.45, y=0.59))
                dev.off()
              }
              if (map_Type == "FSI")
              {
                png(paste(work_Dir, "Map.FSI/PNG/E_", epoch,".FSI_with_Dendrogram.W_(", start_Window,",", end_Window,").T_All.D_", direction,".C_", criterion, ".in_R.png", sep=""), width = 350 * 1.2, height = 130, res =300, units = "mm")
                grid.newpage()
                print(p2, vp=viewport(0.83, 0.2, x=0.457, y=0.095))
                print(p3, vp=viewport(0.1, 0.825, x=0.95, y=0.585))
                print(p1, vp=viewport(0.9, 0.8, x=0.45, y=0.59))
                
                dev.off()
              }
            }
          }
        }
      }
    }
  }
}
