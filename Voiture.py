# -*- coding: utf-8 -*-

#modif. du dossier de travail
import os
os.chdir("D:/PYTHON/Projet ADD")

#librairie pandas
import pandas

#version
print(pandas.__version__)

#chargement de la première feuille de données
X = pandas.read_excel("Jeu_de_donnees_voiture.xlsx",sheet_name=0,header=0,index_col=0)

#dimension
print(X.shape)

#nombre d'observations
n = X.shape[0]

#nombre de variables
p = X.shape[1]

#affichage
print(X)

#scikit-learn
import sklearn

#vérification de la version
print(sklearn.__version__)

#classe pour standardisation
from sklearn.preprocessing import StandardScaler

#instanciation
sc = StandardScaler()

#transformation
Z = sc.fit_transform(X)
print(Z)

#vérification - librairie numpy
import numpy

#moyenne
print(numpy.mean(Z,axis=0))
#ecart-type
print(numpy.std(Z,axis=0,ddof=0))

#classe pour l'ACP
from sklearn.decomposition import PCA

#instanciation
acp = PCA(svd_solver='full')

#affichage des paramètres
print(acp)

#calculs
coord = acp.fit_transform(Z)

#nombre de composantes calculées
print(acp.n_components_)

#variance expliquée
print(acp.explained_variance_)

#valeur corrigée
eigval = (n-1)/n*acp.explained_variance_
print(eigval)

#ou bien en passant par les valeurs singulières
print(acp.singular_values_**2/n)

#proportion de variance expliquée
print(acp.explained_variance_ratio_)


#importation librairie graphique
import matplotlib.pyplot as plt

#scree plot
plt.plot(numpy.arange(1,p+1),eigval)
plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.show()

#cumul de variance expliquée
plt.plot(numpy.arange(1,p+1),numpy.cumsum(acp.explained_variance_ratio_))
plt.title("Explained variance vs. # of factors")
plt.ylabel("Cumsum explained variance ratio")
plt.xlabel("Factor number")
plt.show()

#seuils pour test des bâtons brisés
bs = 1/numpy.arange(p,0,-1)
bs = numpy.cumsum(bs)
bs = bs[::-1]

#test des bâtons brisés
print(pandas.DataFrame({'Val.Propre':eigval,'Seuils':bs}))

#coordonnées factorielles des individus
cf = pandas.DataFrame(coord,index=X.index,columns=numpy.arange(1,p+1))
print(cf)

#positionnement des individus dans le premier plan
fig, axes = plt.subplots(figsize=(30,30))
axes.set_xlim(-6,6)
axes.set_ylim(-6,6)

for i in range(n):
    plt.annotate(X.index[i],(coord[i,0],coord[i,1]),color='r')
    
#ajouter les axes
plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)

#affichage
plt.show()

#contribution des individus dans l'inertie totale
di = numpy.sum(Z**2,axis=1)
print(pandas.DataFrame({'ID':X.index,'d_i':di}))

#qualité de représentation des individus - COS2
cos2 = coord**2
for j in range(p):
    cos2[:,j] = cos2[:,j]/di

print(pandas.DataFrame({'id':X.index,'COS2_1':cos2[:,0],'COS2_2':cos2[:,1]}))

#vérifions la théorie - somme en ligne des cos2 = 1
print(numpy.sum(cos2,axis=1))

#contributions aux axes
ctr = coord**2
for j in range(p):
    ctr[:,j] = ctr[:,j]/(n*eigval[j])
    
print(pandas.DataFrame({'id':X.index,'CTR_1':ctr[:,0],'CTR_2':ctr[:,1]}))

#vérifions la théorie
print(numpy.sum(ctr,axis=0))

#le champ components_ de l'objet ACP
print(acp.components_)

#racine carrée des valeurs propres
sqrt_eigval = numpy.sqrt(eigval)

#corrélation des variables avec les axes
corvar = numpy.zeros((p,p))

for l in range(p):
    corvar[:,l] = acp.components_[l,:]*sqrt_eigval[l]
    
#afficher la matrice des corrélations variables x facteurs    
print(corvar)

#on affiche pour les deux premiers axes
print(pandas.DataFrame({'id':X.columns,'COR_1':corvar[:,0],'COR_2':corvar[:,1]}))

#cercle des corrélations
fig, axes = plt.subplots(figsize=(20,20))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)

for j in range(p):
    plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]),color='b')
    
#ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

#ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)

#affichage
plt.show()
