# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:27:20 2022

@author: Benk
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 09:31:39 2022

@author: Benk
"""
import pandas as pd
from pandasql import sqldf
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#-----------------Belleabeat datat=set-----------------------------
df = pd.read_csv('diabetes.csv')

#------------------------------------------------------------------------------
# ----------------------------Basic Exemple--------------
# raws = {'AA': [21, 15, 13, 21, 0 , 0], 
#       'BB': [21, 15, 13, 21, 0, 0],
#       'CC': [np.nan, np.nan, np.nan, np.nan, 0, 0], 
#       'DD': [4, 24, 31, 2, 0, 0],
#       'EE': [25, 94, 57, 62, 0, 0],
#       'FF': [25, 94, 57, 62, 0, 0],
#       'FF': [25, 94, 57, 62, 0, 0],
#       'GG': [np.nan, np.nan, np.nan, 5, 0, 0],
#       'HH': [np.nan, np.nan, 22, 5, 0, 0],
#       'II': [np.nan, 5, 5, 5, 5, 5],
#       'JJ': [2.5, 2.4, 2.4, 2.5, 2.5, 2.5],
#       'KK': [2, 2, 2, 2, 2, 2],}
# columns = ['A', 'B', 'C', 'D', 'E','F']
# df = pd.DataFrame(raws, columns)

for col in df.columns:
    print(col)
#------------------Removing duplicated features--------------------------------
print('size of the original  dataframe: ',df.shape)
Rdf=df.T.drop_duplicates().T
ListRemovColumns=list(set(df) - set(df))
print('List Of the  duplicated features wich are removed now ')
print(ListRemovColumns)
df=Rdf
print('duplicated features removed, new data size: ',df.shape)

#------------------Removing duplicated samples (The same observation of differents )-----
duplicateObser = df[df.duplicated()]
LabelsDupObser=duplicateObser.axes[0].tolist()
print('Number of duplicated observations:', duplicateObser.shape[0])
print('List of the duplicated Observations', LabelsDupObser)

#------------------Remove missing values---------------------------
ThresholdMissVals=.5
pct_null = df.isnull().sum() / len(df)
missing_features = pct_null[pct_null >= ThresholdMissVals].index
df.drop(missing_features, axis=1, inplace=True)
print('missing values removed, new data size:', df.shape)

#df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
for col in df.columns:
    print(col)

CorrelMat= df.corr()
Choice=1

if Choice==1:
#---------------------------------Graph---------------------------
    G=nx.Graph()
    stocks = CorrelMat.index.values
    CorrelMat = np.asmatrix(CorrelMat)
    CorrelMat[abs(CorrelMat) < .15] = 0

    G = nx.from_numpy_matrix(CorrelMat)
    G = nx.relabel_nodes(G,lambda x: stocks[x])
    G.edges(data=True)
    nx.draw(G,with_labels=True)
    plt.show(G)
#---------------------------------SCATTER---------------------------
if Choice==2:
    
    ax = seaborn.heatmap(
        CorrelMat, 
        vmin=-1, vmax=1, center=0,
        cmap=seaborn.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
      
#---------------------------------CYTOSCAPE---------------------------
if Choice==3:
    CorrelMat= DataSet.corr()
    new_df = CorrelMat.where(np.tril(np.ones(CorrelMat.shape), k=-1).astype(np.bool))
    new_df = pd.DataFrame(new_df.stack().reset_index())
    new_df.columns = ['Row','Column','Value']
#    new_df["Value"].hist(bins=20)
    thres_df = new_df[abs(new_df["Value"])>0.2]
   #thres_df.to_csv('ColonData.tsv',sep='\t')# -*- coding: utf-8 -*-


#-----------Automatically features with  var=0 -----------
# Sdev=df.var(axis=0)
# #hist = Sdev.hist()
# Sdev=Sdev.sort_values()
# NewSdev=Sdev[Sdev.values!=0]
# LabelsNonZero=NewSdev.index.values

# df1=df[LabelsNonZero]
# df=df1
# print('Constant features if there any are remove (Var(Feature-i=0)),  new df size:', df.shape)
#----------------------------------------Filling The missing values with 0-----------------
# Fill NaN values with zero
# df=df.fillna(0)
