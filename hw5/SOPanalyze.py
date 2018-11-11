import numpy as np
from sklearn.cluster import KMeans
import sklearn.cluster
from matplotlib import mlab
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.naive_bayes import GaussianNB

## Load data
loader = np.load('SOPdata.npz')
from scipy.sparse import csr_matrix 
X = csr_matrix(( loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
vocab = loader['vocab']
tias = loader['tias']



def plot_bar_from_counter(counter, ax=None):
    """"
    This function creates a bar plot from a counter.

    :param counter: This is a counter object, a dictionary with the item as the key
     and the frequency as the value
    :param ax: an axis of matplotlib
    :return: the axis wit the object in it
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    frequencies = counter.values()
    names = counter.keys()

    x_coordinates = np.arange(len(counter))
    ax.bar(x_coordinates, frequencies, align='center')

    ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(names))

    return ax

## Build topic model using NMF
from sklearn import decomposition
num_topics = 6
num_top_words = 15

clf = decomposition.NMF(n_components=num_topics, random_state=1)

# this next step may take some time
doctopic = clf.fit_transform(X)
doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)

# print words associated with topics
topic_words = []
for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])

for t in range(len(topic_words)):
    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))



kmeans = KMeans(n_clusters=11, random_state=0).fit(doctopic)
clusters = kmeans.predict(doctopic)
clusters = clusters.reshape(-1,1)

centroid = kmeans.cluster_centers_
clusterid = kmeans.labels_

doctopic_pca = mlab.PCA(doctopic)
cutoff = doctopic_pca.fracs[1]
doctopic_2d = doctopic_pca.project(doctopic, minfrac=cutoff)
centroid_2d = doctopic_pca.project(centroid, minfrac=cutoff)
colors = ['red', 'green', 'blue', 'yellow', 'black', 'cyan', 'magenta', 'brown', 'tomato', 'c', 'slateblue']
plt.figure()
plt.xlim([doctopic_2d[:,0].min() - 0.5, doctopic_2d[:,0].max() + 0.5])
plt.ylim([doctopic_2d[:,1].min() - 0.5, doctopic_2d[:,1].max() + 0.5])
plt.xticks([], []); plt.yticks([], [])

plt.scatter(centroid_2d[:,0], centroid_2d[:,1], marker='o', c=colors, s=100)

for i,((x,y), kls) in enumerate(zip(doctopic_2d, clusterid)):
    plt.annotate(str(i), xy=(x,y), xytext=(0,0), textcoords='offset points', color=colors[kls])
plt.show()

print(set(tias))

print(kmeans.labels_)
print(kmeans.cluster_centers_)



print(tias.shape)
print(clusters.shape)

#from sklearn.naive_bayes import GaussianNB
s= tias
mydict={}
i = 0
for item in s:
    if(i>0 and item in mydict):
        continue
    else:    
       i = i+1
       mydict[item] = i
k=[]
for item in s:
    k.append(mydict[item])
plt.figure(1)
plt.hist(k)
plt.title("Technical Interest Area distribution")
plt.show()

plt.figure(2)
plt.hist(clusters)  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()

gip = GaussianNB()
train_X = doctopic[1:1000]
train_y = k[1:1000]
y_pred = gip.fit(train_X,train_y).predict(doctopic)
print("Number of mislabeled points out of a total %d points : %d" % (doctopic.shape[0],(k != y_pred).sum()))
print("Classifier Naive Bayes")
