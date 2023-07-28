# Margot Zhao

import pandas as pd
import matplotlib.pyplot as plt

# (1) read data into a dataframe
wine = pd.read_csv("wineQualityReds.csv")

# (2) drop "wine" from the dataframe
wine.drop(columns="Wine", inplace=True)

# (3) extract quality and store it in another variable
qual = wine["quality"]
# (4) drop quality from the dataframe
wine.drop(columns="quality", inplace=True)

# (5) print dataframe and quality
print("Dataframe:", wine)
print("Quality:\n", qual)
print()

# (6) normalize all columns of the data frame
from sklearn.preprocessing import Normalizer
norm = Normalizer()
normwine = norm.fit_transform(wine)

# turn normwine array back into a dataframe
normwineDF = pd.DataFrame(normwine, columns=["fixed.acidity", "volatile.acidity", "citric.acid",
                          "residual.sugar", "chlorides", "free.sulfur.dioxide", "total.sulfur.dioxide",
                          "density", "pH", "sulphates", "alcohol"])

# (7) print normalized data
print("Normalized Dataframe:\n", normwineDF)

# (8) create a range of k values 1 through 11
from sklearn.cluster import KMeans
krange = range(1,11)
# create empty list for inertia
inertia = []
# iterate through krange and store values in "inertia" list
for k in krange:
        model = KMeans(n_clusters=k)
        model.fit(normwineDF)
        # append inertia
        inertia.append(model.inertia_)

# (9) plot the inertia as a function of the number of clusters
plt.plot(krange, inertia, "-o")
# label axis
plt.xlabel("# of Clusters")
plt.ylabel("Inertia")
# set x ticks
plt.xticks(krange)
# display plot
plt.show()

# (10) I would pick 6 for k

# (11) cluster wine into k-clusters
# instantiate, use random_state = 2021
model = KMeans(n_clusters=6, random_state=2021)
# assign cluster number to each wine
model.fit(normwineDF)
# print dataframe w/ cluster number for each wine
normwineDF["Cluster"] = pd.Series(model.labels_)
print(normwineDF)

# (12) add quality back to the dataframe
normwineDF["Quality"] = qual

# (13) pandas crosstab of cluster # v.s. quality
crosstab = pd.crosstab(normwineDF["Quality"], normwineDF["Cluster"])
print(crosstab)
# across all clusters, it seems like most of the wine has quality rating between 5-7
# compared to other clusters, cluster 4 has a more wines with a quality rating of 5, and less of wine rated 6 or 7
# clusters 0, 1, and 2 seems fairly similar in their distribution of wine quality ratings
