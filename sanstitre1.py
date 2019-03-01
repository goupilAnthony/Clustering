# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:37:15 2019

@author: antho

TP clustering 

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%

data1 = pd.read_csv('dataset_1.csv')
X= data1.iloc[:,[1,2]]
data1.columns
from sklearn.cluster import KMeans

kmeanModel = KMeans(n_clusters=2).fit(X)
kmeanModel.fit(X)

labels = kmeanModel.labels_
data1 = pd.DataFrame(data1,columns=['normalized_age', 'normalized_salary','Cluster_label'])
data1['Cluster_label'] = labels

sns.scatterplot(data1.iloc[:,0],data1.iloc[:,1],hue=data1.iloc[:,2])

kmeanModel.inertia_

from scipy.spatial.distance import cdist

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow curve to find the right number of clusters
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#%%
data2 = pd.read_csv('dataset_2.csv')
#sns.scatterplot(data2.iloc[:,1],data2.iloc[:,2])

X= data2.iloc[:,[1,2]]

distortions = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow curve to find the right number of clusters
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


kmeanModel = KMeans(n_clusters=20).fit(X)
kmeanModel.fit(X)

labels = kmeanModel.labels_
data2 = pd.DataFrame(data2,columns=['normalized_age', 'normalized_salary','Cluster_label'])
data2['Cluster_label'] = labels

sns.scatterplot(data2.iloc[:,0],data2.iloc[:,1],hue=data2.iloc[:,2])

#%%

def plotModels():
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import SpectralClustering
    from sklearn.mixture import GaussianMixture
    
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, sharex='col', sharey='row')
    ax1.set_title('Dataset_2')
    ax2.set_title('Dataset_3')
    dico = {'kmeans':[3,3],'dbeps':[0.150,0.250],'dbmin':[5,2],'spectral':[3,2],'gaussian':[3,2]}
    
    for i in range(0,2):
        if i == 0:
            data = pd.read_csv('dataset_2.csv')
        else:
            data = pd.read_csv('dataset_3.csv')
        
        
        X= data.iloc[:,[1,2]].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    
        kmeanModel = KMeans(n_clusters=dico['kmeans'][i]).fit(X)
        kmeanModel.fit(X_scaled)
        labels = kmeanModel.labels_
        data = pd.DataFrame(X_scaled,columns=['normalized_age', 'normalized_salary'])
        data['Cluster_label'] = labels
        
        if i == 0:
            ax1.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,2],cmap='plasma')
            ax1.set_ylabel('KMeans')
        else:
            ax2.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,2],cmap='plasma')
        
        
    
        dbscan = DBSCAN(eps=dico['dbeps'][i], min_samples = dico['dbmin'][i])
        clusters = dbscan.fit_predict(X_scaled)
    
        data = pd.DataFrame(X_scaled,columns=['normalized_age', 'normalized_salary'])
        data['Cluster_label'] = clusters
        
        if i == 0:
            ax3.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,2],cmap='plasma')
            ax3.set_ylabel('DBscan')
        else:
            ax4.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,2],cmap='plasma')
        
     
    
    
        clustering = SpectralClustering(n_clusters=dico['spectral'][i], assign_labels="discretize",gamma=0.5)
        clustering.fit(X_scaled)
        labels = clustering.labels_
        data = pd.DataFrame(X_scaled,columns=['normalized_age', 'normalized_salary'])
        data['Cluster_label'] = labels
        
        if i == 0:
            ax5.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,2],cmap='plasma')
            ax5.set_ylabel('Spectral Clustering')
        else:
            ax6.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,2],cmap='plasma')
        
        
        gmm = GaussianMixture(n_components=dico['gaussian'][i]).fit(X)
        labels = gmm.predict(X)
        data = pd.DataFrame(X_scaled,columns=['normalized_age', 'normalized_salary'])
        data['Cluster_label'] = labels
    
        if i == 0:
            ax7.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,2],cmap='plasma')
            ax7.set_ylabel('Gaussian Mixture')
        else:
            ax8.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,2],cmap='plasma')
    
    plt.show()



plotModels()

dico = {'kmeans':[3,3],'dbeps':[0.150,0.250],'dbmin':[5,2],'spectral':[3,2],'gaussian':[3,2]}
from sklearn.mixture import BayesianGaussianMixture
gmm = BayesianGaussianMixture(n_components=dico['gaussian'][1]).fit(X)
labels = gmm.predict(X)
data = pd.DataFrame(X_scaled,columns=['normalized_age', 'normalized_salary'])
data['Cluster_label'] = labels

plt.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,2],cmap='plasma')




