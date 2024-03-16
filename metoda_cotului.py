from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils
import pickle
import os 


with open(r'D:\Facultate\Licenta\Codul sursa\obiecte\imagini.pickle', 'rb') as f:
   imagini = pickle.load(f)

y_sse_list = []
#iterarea prin fiecare imagine pentru determinarea numarului optim de centroizi pentru fiecare imagine
for lot_imagini in imagini:
    for i, img in enumerate(lot_imagini):
        if i%3 == 0:
            # calcularea valorilor SSE pentru diferite valori ale k
            sse = []
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
                kmeans.fit(img)
                sse.append(kmeans.inertia_)
            
            y_sse_list.append(sse)
            
        else:
            pass

# creare figura și axele
fig, ax = plt.subplots()
# plasare fiecare grafic in axe
for y in y_sse_list:
    ax.plot(range(1, 11), y)

plt.show()

# vizionare lista salvata in optim_centroizi
with open(r'D:\Facultate\Licenta\Codul sursa\obiecte\metoda_cotului\metoda_cotului_rezultate.pickle', 'rb') as f:
    a = pickle.load(f)
 
# absolute_path = os.path.dirname(__file__)