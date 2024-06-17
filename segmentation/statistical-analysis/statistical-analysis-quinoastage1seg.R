###########################################################################################
##############ANALYSIS FOR SIGNIFFICANCE AND LSMEANS OF AP RESULT MAIZE-PERU-200 and MAIZE-PERU-1000###########
#############################LOGIT TRANSFORMATION##########################################

#######important to add these results to paper###########
#no significant interaction of datasetsize and use of minimask --> no minimask always better!!

##################
##load libraries##
##################
library(dplyr)
#for including TypeIII error
library(car)

#for lsmeans
library(emmeans)

#for backtransformation inv.logit
library(GMCM)

###########################
##read in data and modify##
###########################

#set working directory
setwd("~/Desktop/quino-img-analysis/2-stage-approach/1-segmentation-part/statistical-analysis")

#read in data
param<- readxl::read_xlsx("quinoa-panicles-2stage-segm-model-param.xlsx", sheet=2)

#make as factor
str(param)
param$ap595<-as.numeric(param$ap595)
param$ap595<-param$ap595*100
param$ap595_logit<-log(param$ap595)


###########################
#########Model Selection###
###########################

#without logit - residual plots okay
options(contrasts=c(unordered="contr.sum",ordered="contr.poly"))

Anova(step1 <- lm(ap595~ loss + heads.m + mask_resolution + loss:heads.m + loss:mask_resolution + heads.m:mask_resolution  ,
                         data=param),type="III",singular.ok = TRUE)
plot(step1)


#dropping ns factors
options(contrasts=c(unordered="contr.sum",ordered="contr.poly"))

Anova(step2 <- lm(ap595~heads.m + mask_resolution, data=param),type="III",singular.ok = TRUE)
plot(step2)





###########################
#########Lsmeans and contrasts###########
###########################

#get lsmeans dataset:model
res1<-CLD(emmeans(step2,pairwise  ~ heads.m))
write.table(res1,file="lsmeans-heads.m.csv",sep=";", dec=",",
            col.names=NA, row.names=TRUE,qmethod="double")

#get contrasts mask_resolution
res2<-CLD(emmeans(step2,pairwise  ~ mask_resolution))
write.table(res2,file="lsmeans-mask_resolution.csv",sep=",", dec=".",
            col.names=NA, row.names=TRUE,qmethod="double")

###########################
#########variance components###########
###########################
model<-lme4::lmer(ap595~1 + (1|heads.m) + (1|mask_resolution), data=param)
lme4::VarCorr(model)



###########################
#########for reporting, calculate mean ap595 per model###########
###########################

summarizeddata<-param %>% 
  group_by (modeltype) %>% 
  summarise (ap595_mean=mean(ap595)) %>% 
  arrange(desc(ap595_mean))
writexl::write_xlsx(summarizeddata,"quinoastage1seg_summarized_data_by_model_ap595.xlsx")
