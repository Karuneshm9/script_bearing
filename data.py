#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
from scipy.io import loadmat
import scipy.io as sio
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import model_evaluation_utils as meu
#from graphviz import Source
from sklearn import tree 
from IPython.display import Image
#from skater.core.explanations import Interpretation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor

from skater.model import InMemoryModel


# In[6]:


# Decreasing then increasing speed
file='H-D-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df1=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df1= pd.DataFrame(newdata,columns=cols)
df1=df1.astype(float)
df1['Status']='Healthy'
#print(df1)


# In[3]:


file='H-D-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df2=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df2= pd.DataFrame(newdata,columns=cols)
df2=df2.astype(float)
df2['Status']='Healthy'
#print(df2)


# In[4]:


file='H-D-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
file=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df3= pd.DataFrame(newdata,columns=cols)
df3=df3.astype(float)
df3['Status']='Healthy'
#print(df3)


# In[5]:


file='I-D-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df4=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df4= pd.DataFrame(newdata,columns=cols)
df4=df4.astype(float)
df4['Status']='fault(inner race fault)'
#print(file)


# In[6]:


file='I-D-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df5=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df5= pd.DataFrame(newdata,columns=cols)
df5=df5.astype(float)
df5['Status']='fault(inner race fault)'
#print(df5)


# In[7]:


file='I-D-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df6=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df6= pd.DataFrame(newdata,columns=cols)
df6=df6.astype(float)
df6['Status']='fault(inner race fault)'
#print(df6)


# In[8]:


file='O-D-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df7=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df7= pd.DataFrame(newdata,columns=cols)
df7=df7.astype(float)
df7['Status']='fault(outer race fault)'
#print(df7)


# In[9]:


file='O-D-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df8=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df8= pd.DataFrame(newdata,columns=cols)
df8=df8.astype(float)
df8['Status']='fault(outer race fault)'
#print(df8)


# In[10]:


file='O-D-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df9=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df9= pd.DataFrame(newdata,columns=cols)
df9=df9.astype(float)
df9['Status']='fault(outer race fault)'
#print(df9)

file='B-D-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df10=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df10= pd.DataFrame(newdata,columns=cols)
df10=df10.astype(float)
df10['Status']='Ball fault'
#print(file)


# In[6]:


file='B-D-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df11=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df11= pd.DataFrame(newdata,columns=cols)
df11=df11.astype(float)
df11['Status']='Ball fault'
#print(df5)


# In[7]:


file='B-D-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df12=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df12= pd.DataFrame(newdata,columns=cols)
df12=df12.astype(float)
df12['Status']='Ball fault'
#print(df6)


# In[8]:


file='C-D-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df13=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df13= pd.DataFrame(newdata,columns=cols)
df13=df13.astype(float)
df13['Status']='combined fault'
#print(df7)


# In[9]:


file='C-D-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df14=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df14= pd.DataFrame(newdata,columns=cols)
df14=df14.astype(float)
df14['Status']='combined fault'
#print(df8)


# In[10]:


file='C-D-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df15=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df15= pd.DataFrame(newdata,columns=cols)
df15=df15.astype(float)
df15['Status']='Combined fault'
# In[11]:


DnI_list=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15]
DnI=pd.concat(DnI_list)
con_list=[DnI['Status'].str.contains('Healthy',na=False),DnI['Status'].str.contains("fault",na=False)]
val_list=[0,1]
DnI['Result']=np.select(con_list,val_list)
DnI.columns=['DnI_1','DnI_2','DnI_3','DnI_4']
#print(df)
#outdata=df.to_dict('list')
#sio.savemat('DnI.mat',outdata)
#print(loadmat('DnI.mat'))


# In[12]:


# In[ ]:


# increasing speed
file='H-A-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df1=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df1= pd.DataFrame(newdata,columns=cols)
df1=df1.astype(float)
df1['Status']='Healthy'
#print(df1)


# In[3]:


file='H-A-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df2=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df2= pd.DataFrame(newdata,columns=cols)
df2=df2.astype(float)
df2['Status']='Healthy'
#print(df2)


# In[4]:


file='H-A-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
file=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df3= pd.DataFrame(newdata,columns=cols)
df3=df3.astype(float)
df3['Status']='Healthy'
#print(df3)


# In[5]:


file='I-A-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df4=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df4= pd.DataFrame(newdata,columns=cols)
df4=df4.astype(float)
df4['Status']='Fault(inner race fault)'
#print(file)


# In[6]:


file='I-A-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df5=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df5= pd.DataFrame(newdata,columns=cols)
df5=df5.astype(float)
df5['Status']='Faulty(inner race fault)'
#print(df5)


# In[7]:


file='I-A-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df6=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df6= pd.DataFrame(newdata,columns=cols)
df6=df6.astype(float)
df6['Status']='Faulty(inner race fault)'
#print(df6)


# In[8]:


file='O-A-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df7=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df7= pd.DataFrame(newdata,columns=cols)
df7=df7.astype(float)
df7['Status']='Faulty(outer race fault)'
#print(df7)


# In[9]:


file='O-A-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df8=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df8= pd.DataFrame(newdata,columns=cols)
df8=df8.astype(float)
df8['Status']='Faulty(outer race fault)'
#print(df8)


# In[10]:


file='O-A-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df9=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df9= pd.DataFrame(newdata,columns=cols)
df9=df9.astype(float)
df9['Status']='Faulty(outer race fault)'
#print(df9)

file='B-A-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df10=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df10= pd.DataFrame(newdata,columns=cols)
df10=df10.astype(float)
df10['Status']='Ball fault'
#print(file)


# In[6]:


file='B-A-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df11=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df11= pd.DataFrame(newdata,columns=cols)
df11=df11.astype(float)
df11['Status']='Ball Fault'
#print(df5)


# In[7]:


file='B-A-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df12=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df12= pd.DataFrame(newdata,columns=cols)
df12=df12.astype(float)
df12['Status']='Ball Fault'
#print(df6)


# In[8]:


file='C-A-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df13=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df13= pd.DataFrame(newdata,columns=cols)
df13=df13.astype(float)
df13['Status']='combined fault'
#print(df7)


# In[9]:


file='C-A-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df14=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df14= pd.DataFrame(newdata,columns=cols)
df14=df14.astype(float)
df14['Status']='combined fault'
#print(df8)


# In[10]:


file='C-A-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df15=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df15= pd.DataFrame(newdata,columns=cols)
df15=df15.astype(float)
# In[11]:


df_list=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15]
IS=pd.concat(df_list)
con_list=[IS['Status'].str.contains('Healthy',na=False),IS['Status'].str.contains("Faulty",na=False)]
val_list=[0,1]
IS['Result']=np.select(con_list,val_list)
IS.columns=['IS_1','IS_2','IS_3','IS_4']
#print(df)
#outdata=df.to_dict('list')
#sio.savemat('DnI.mat',outdata)
#print(loadmat('DnI.mat'))


# In[12]:


# In[8]:


# decreasing speed
file='H-B-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df1=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df1= pd.DataFrame(newdata,columns=cols)
df1=df1.astype(float)
df1['Status']='Healthy'
#print(df1)


# In[3]:


file='H-B-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df2=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df2= pd.DataFrame(newdata,columns=cols)
df2=df2.astype(float)
df2['Status']='Healthy'
#print(df2)


# In[4]:


file='H-B-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
file=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df3= pd.DataFrame(newdata,columns=cols)
df3=df3.astype(float)
df3['Status']='Healthy'
#print(df3)


# In[5]:


file='I-B-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df4=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df4= pd.DataFrame(newdata,columns=cols)
df4=df4.astype(float)
df4['Status']='Fault(inner race fault)'
#print(file)


# In[6]:


file='I-B-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df5=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df5= pd.DataFrame(newdata,columns=cols)
df5=df5.astype(float)
df5['Status']='Fault(inner race fault)'
#print(df5)


# In[7]:


file='I-B-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df6=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df6= pd.DataFrame(newdata,columns=cols)
df6=df6.astype(float)
df6['Status']='Fault(inner race fault)'
#print(df6)


# In[8]:


file='O-B-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df7=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df7= pd.DataFrame(newdata,columns=cols)
df7=df7.astype(float)
df7['Status']='Fault(outer race fault)'
#print(df7)


# In[9]:


file='O-B-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df8=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df8= pd.DataFrame(newdata,columns=cols)
df8=df8.astype(float)
df8['Status']='Fault(outer race fault)'
#print(df8)


# In[10]:


file='O-B-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df9=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df9= pd.DataFrame(newdata,columns=cols)
df9=df9.astype(float)
df9['Status']='Fault(outer race fault)'
#print(df9)

file='B-B-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df10=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df10= pd.DataFrame(newdata,columns=cols)
df10=df10.astype(float)
df10['Status']='Ball fault'
#print(file)


# In[6]:


file='B-B-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df11=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df11= pd.DataFrame(newdata,columns=cols)
df11=df11.astype(float)
df11['Status']='Ball Fault'
#print(df5)


# In[7]:


file='B-B-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df12=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df12= pd.DataFrame(newdata,columns=cols)
df12=df12.astype(float)
df12['Status']='Ball Fault'
#print(df6)


# In[8]:


file='C-B-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df13=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df13= pd.DataFrame(newdata,columns=cols)
df13=df13.astype(float)
df13['Status']='combined fault'
#print(df7)


# In[9]:


file='C-B-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df14=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df14= pd.DataFrame(newdata,columns=cols)
df14=df14.astype(float)
df14['Status']='combined fault'
#print(df8)


# In[10]:


file='C-B-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df15=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df15= pd.DataFrame(newdata,columns=cols)
df15=df15.astype(float)
# In[11]:


df_list=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15]
DS=pd.concat(df_list)
con_list=[DS['Status'].str.contains('Healthy',na=False),DS['Status'].str.contains("Faulty",na=False)]
val_list=[0,1]
DS['Result']=np.select(con_list,val_list)
DS.columns=['DS_1','DS_2','DS_3','DS_4']
#print(df)
#outdata=df.to_dict('list')
#sio.savemat('DnI.mat',outdata)
#print(loadmat('DnI.mat'))


# In[12]:


# In[ ]:


# decreasing speed
file='H-C-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df1=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df1= pd.DataFrame(newdata,columns=cols)
df1=df1.astype(float)
df1['Status']='Healthy'
#print(df1)


# In[3]:


file='H-C-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df2=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df2= pd.DataFrame(newdata,columns=cols)
df2=df2.astype(float)
df2['Status']='Healthy'
#print(df2)


# In[4]:


file='H-C-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
file=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df3= pd.DataFrame(newdata,columns=cols)
df3=df3.astype(float)
df3['Status']='Healthy'
#print(df3)


# In[5]:


file='I-C-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df4=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df4= pd.DataFrame(newdata,columns=cols)
df4=df4.astype(float)
df4['Status']='Faulty(inner race fault)'
#print(file)


# In[6]:


file='I-C-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df5=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df5= pd.DataFrame(newdata,columns=cols)
df5=df5.astype(float)
df5['Status']='Faulty(inner race fault)'
#print(df5)


# In[7]:


file='I-C-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df6=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df6= pd.DataFrame(newdata,columns=cols)
df6=df6.astype(float)
df6['Status']='Faulty(inner race fault)'
#print(df6)


# In[8]:


file='O-C-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df7=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df7= pd.DataFrame(newdata,columns=cols)
df7=df7.astype(float)
df7['Status']='Faulty(outer race fault)'
#print(df7)


# In[9]:


file='O-C-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df8=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df8= pd.DataFrame(newdata,columns=cols)
df8=df8.astype(float)
df8['Status']='Faulty(outer race fault)'
#print(df8)


# In[10]:


file='O-C-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df9=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df9= pd.DataFrame(newdata,columns=cols)
df9=df9.astype(float)
df9['Status']='Faulty(outer race fault)'
#print(df9)

file='B-C-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df10=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df10= pd.DataFrame(newdata,columns=cols)
df10=df10.astype(float)
df10['Status']='Ball fault'
#print(file)


# In[6]:


file='B-C-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df11=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df11= pd.DataFrame(newdata,columns=cols)
df11=df11.astype(float)
df11['Status']='Ball Fault'
#print(df5)


# In[7]:


file='B-C-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df12=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df12= pd.DataFrame(newdata,columns=cols)
df12=df12.astype(float)
df12['Status']='Ball Fault'
#print(df6)


# In[8]:


file='C-C-1.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df13=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df13= pd.DataFrame(newdata,columns=cols)
df13=df13.astype(float)
df13['Status']='combined fault'
#print(df7)


# In[9]:


file='C-C-2.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df14=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df14= pd.DataFrame(newdata,columns=cols)
df14=df14.astype(float)
df14['Status']='combined fault'
#print(df8)


# In[10]:


file='C-C-3.mat'

annots=loadmat(file)
newdata=[]
cols=list()
for key,grp in annots.items():
        
        
    #print(grp,key)
        newdata.append(grp)
    #print(grp)
        cols.append(key)
cols=cols[-2:]
newdata=newdata[-2:]
file=file[:-4].replace('-','')
k=file
df15=pd.DataFrame()
newdata=list(zip(newdata[0],newdata[1]))
df15= pd.DataFrame(newdata,columns=cols)
df15=df15.astype(float)
# In[11]:


df_list=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15]
IDS=pd.concat(df_list)
con_list=[IDS['Status'].str.contains('Healthy',na=False),IDS['Status'].str.contains("Faulty",na=False)]
val_list=[0,1]
IDS['Result']=np.select(con_list,val_list)
IDS.columns=['IDS_1','IDS_2','IDS_3','IDS_4']
#print(df)
#outdata=df.to_dict('list')
#sio.savemat('DnI.mat',outdata)
#print(loadmat('DnI.mat'))


# In[12]:

