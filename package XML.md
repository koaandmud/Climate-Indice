Bad case for readHTMLTable function of package XML

In the previous article, the readHTMLTable function worked perfectly to get the PM 2.5 data frame from a web page of Ibaraki prefecture.  So then I tried another web, a page of Chiba prefecture, to get the PM 2.5 data.  However that failed.

Here I show a bad case of the readHTMLTable function, and introduce how to handle such a troublesome html pages.
Fig. 1. Chiba Prefecture shows hourly air pollution data on the web

Fig. 1. Chiba Prefecture shows hourly air pollution data on the web
Case 1: Html frame

library(XML)
tables <- readHTMLTable('http://www.taiki.pref.chiba.lg.jp/sokuho.html')
str(tables)

# Result
> str(tables)
Named list()

No tables were extracted.  Why?

To see what is happening, reading the raw html source is a good way. The url() function is included in the base system of R, that function opens a connection to a url. Or using the function getURL in the CRAN package RCurl may be  easier.

con <- url('http://www.taiki.pref.chiba.lg.jp/sokuho.html')
chibasrc <- readLines(con)
close(con)

install.packages(RCurl)
library(RCurl)
chibasrc <- getURL('http://www.taiki.pref.chiba.lg.jp/sokuho.html')

# Result of url()
> chibasrc
[1] "<HTML>"
[2] "<HEAD>"
[3] "<TITLE>\x90\xe7\x97t\x8c\xa7\x82̑\xe5\x8bC\x8a\u008b\xab  \x8e\x9e\x95\xf1</TITLE>"
[4] "<LINK  REL=\"stylesheet\"  HREF=\"default.css\"  TYPE=\"text/css\"  />"
[5] "</HEAD>"
[6] ""
[7] "<FRAMESET  ROWS=\"185,*\"    frameborder=\"no\"  border=\"0\">"
[8] "    <FRAME  SRC=\"sokuho_head.html\"  NAME=\"sokuho_head\"    resize>"
[9] "    <FRAME  SRC=\"sokuho_body.html\"  NAME=\"sokuho_body\"    resize>"
[10] "</FRAMESET>"
[11] "</HTML>"

This source contains no tables.  So, the readHTMLTable function was working correctly. I must give the function inner content’s url instead of the outer frame url.

Also, the kanji text cannot read out. This occurs when the web server does not send a correct header to judge the encoding of its content.
Case 2: Encoding

I told these functions to encode the page as “Shift_JIS”, and got correct kanjis. Though this is as same as the default encoding “cp932” for the R 2.15 for Japanese Windows, the R 2.15 for Japanese Mac OS uses “UTF-8” for the default.

con <- url('http://www.taiki.pref.chiba.lg.jp/sokuho.html', encoding="shift_jis")
chibasrc <- readLines(con)
close(con)

install.packages(RCurl)
library(RCurl)
chibasrc <- iconv(getURL('http://www.taiki.pref.chiba.lg.jp/sokuho.html', .encoding='shift_jis'), from='shift_jis', to='utf-8')

Now, back to the extraction of tables.

tables.head <- readHTMLTable('http://www.taiki.pref.chiba.lg.jp/sokuho_head.html', stringsAsFactors=F)
tables.body <- readHTMLTable('http://www.taiki.pref.chiba.lg.jp/sokuho_body.html', stringsAsFactors=F)
str(tables.head)
str(tables.body)

# Result
> str(tables.head, max.level=1)
List of 2
$ NULL:'data.frame':    2 obs. of  3 variables:
$ NULL:'data.frame':    2 obs. of  18 variables:
> str(tables.body, max.level=1)
List of 1
$ NULL:'data.frame':    143 obs. of  18 variables:

It worked.

The tables.body has one table with 143 rows. This must be the air pollution data I want. How about the tables.head? The 2nd table of the tables.head has 18 columns that is equivalent to the body table.
Let’s check that.
Fig. 2. Header part and body part of the html table

Fig. 2. Header part and body part of the html table

Figure 2 shows that the tables.head contains column names, and the tables.body contains data rows.
Case 3: Html tables are splitted

Because the column name and the data are located in different tables, I must combine the column name into the data frame.

chiba.head <- tables.head[[2]][1,]
chiba.body <- tables.body[[1]]
names(chiba.body) <- chiba.head
str(chiba.body)

Fig. 3. Columns shifted at chiba.body data frame

Fig. 3. Columns shifted at chiba.body data frame

It seems the columns data have unmatched data types. Columns shifted at some rows.
Case 4: Html table has rowspan attributes

Unfortunately, the readHTMLTable function does not handle rowspan and colspan attributes.  So, the data go into wrong columns.  This is quite wrong situation. Though solving this at parsing time is better, I adjusted the data after parsing at this time.  Because adjusting the parser requires much more development time.

In this table, the rowspan locates only at the 3rd column. Because the parser ignores this attributes, some rows have a shorter columns length. This generates an NA value at the last column. I am going to check these NAs to adjust the column location.

narows <- is.na(chiba.body[,18])

for(i in 18:4) chiba.body[narows,i] <- chiba.body[narows,i-1]
chiba.body[narows,3] <- NA
for(i in 1L:length(chiba.body[,3])) if(is.na(chiba.body[i,3])) chiba.body[i,3] <- chiba.body[i-1,3]
for(i in c(4:14,16:18)) chiba.body[,i] <- as.numeric(chiba.body[,i])

pm.2.5.chiba <- chiba.body[!is.na(chiba.body[,14]),c(1,14)]
pm.2.5.chiba[,1] = as.factor(as.character(pm.2.5.chiba[,1]))

str(chiba.body)
str(pm.2.5.chiba)

Fig. 4. Column adjusted chiba.body

Fig. 4. Column adjusted chiba.body
Fig. 5. pm.2.5.chiba

Fig. 5. pm.2.5.chiba

Done?
No.

The pm.2.5.chiba table has 34 rows, but the 1st column has  33 factor levels. There are duplicate value of the location name.
Case 5:  The identifiers are not unique

Unfortunately, this table does not have a unique identifier. The 1st column has a name of location, and it is expected to act as a unique key. But it does not.

pm.2.5.chiba[duplicated(pm.2.5.chiba[,1]),1]

Fig. 6. Check the duplicate name

Fig. 6. Check the duplicate name

The name “Chiba Masuna” is duplicated. Usually, in such case, a quick way to generate a unique identifier is using row numbers. The following would generate a unique identifier.

paste(rownames(pm.2.5.chiba), pm.2.5.chiba[,1])

Fig. 7. Generated unique identifier

Fig. 7. Generated unique identifier

pm.2.5.chiba[,1] <- as.factor(paste(rownames(pm.2.5.chiba), pm.2.5.chiba[,1]))

Fig. 8. Completed data frame of PM 2.5

Fig. 8. Completed data frame of PM 2.5

Now, I can draw a chart.

quartzFonts(HiraMaru=quartzFont(rep("HiraMaruProN-W4", 4)))
par(family="HiraMaru")
barplot(pm.2.5.chiba[,2], names.arg=pm.2.5.chiba[,1], cex.names=0.64, las=2, col='tomato')

Fig. 9. PM 2.5 come from China to Chiba, Japan

Fig. 9. PM 2.5 come from China to Chiba, Japan

Of course, it can be combined with the result of Ibaraki.

pm.2.5.chibaraki <- rbind(pm.2.5.ibaraki, pm.2.5.chiba)
barplot(pm.2.5.chibaraki[,2], names.arg=pm.2.5.chibaraki[,1], cex.names=0.64, las=2, col=c(rep('springgreen', nrow(pm.2.5.ibaraki)), rep('tomato', nrow(pm.2.5.chiba))))
mtext('Ibaraki', side=3, at=nrow(pm.2.5.ibaraki) %/% 2, col='green', cex=3, padj=1)
mtext('Chiba', side=3, at=nrow(pm.2.5.ibaraki) + nrow(pm.2.5.chiba) %/% 2, col='tomato', cex=3, padj=1)

Fig. 10. PM 2.5 come from China to Chiba and Ibaraki, Japan

Fig. 10. PM 2.5 come from China to Chiba and Ibaraki, Japan
