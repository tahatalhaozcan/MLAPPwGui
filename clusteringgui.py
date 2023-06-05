
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score,calinski_harabasz_score
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk


class ClusteringModels:
    def __init__(self, model_type, n_cluster=None, max_iter=None, distance=None, sample=None, metric=None):
        self.model_type = model_type
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.distance = distance
        self.sample = sample
        self.metric = metric
        self.model = None

    def load_data(self, filename, independent_cols):
       data = None
       valid_file = False

       while not valid_file:
           try:
               data = pd.read_csv(filename)
               valid_file = True
           except:
               try:
                   data = pd.read_excel(filename)
                   valid_file = True
               except:
                   try:
                       data = pd.read_table(filename)
                       valid_file = True
                   except:
                       messagebox.showerror("Hata", "Dosya okunamadı!")
                       filename = filedialog.askopenfilename(
                           filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")],
                           title="Geçerli bir dosya seçin"
                       )

       cols = list(data.columns)
       
       X = data[independent_cols].values
       scaler = StandardScaler()
       X = scaler.fit_transform(X)
       return X,cols
       

    def clustering_model(self):
        if self.model_type == "kmeans":
            if self.n_cluster is not None and self.max_iter is not None:
                self.model = KMeans(n_clusters=self.n_cluster, max_iter=self.max_iter, init="k-means++")
            else:
                raise ValueError("Lütfen Küme Sayısını ve/veya İterasyon sayısını giriniz")
        elif self.model_type == "agc":
            if self.n_cluster is not None:
                self.model = AgglomerativeClustering(n_clusters=self.n_cluster, affinity="euclidean")
            else:
                raise ValueError("Lütfen Küme Sayısını giriniz!")
        elif self.model_type == "dbscan":
            if self.distance is not None and self.sample is not None and self.metric is not None:
                self.model = DBSCAN(eps=self.distance, min_samples=self.sample, metric=self.metric)
            else:
                raise ValueError("Lütfen Mesafe ve/veya Minimum Örnek ve/veya Metrik değerlerini giriniz")
        else:
            return None

    def train_model(self, X_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        self.model.fit(X_train)

    def evaluate_model(self, X_test):
        if self.model_type == "kmeans":
            y_pred = self.model.predict(X_test)
            score = silhouette_score(X_test, y_pred)
            messagebox.showinfo("Sonuç", "Silhouette Score: {:.2f}".format(score))
        elif self.model_type == "agc":
            y_pred = self.model.fit_predict(X_test)
            score = silhouette_score(X_test, y_pred)
            messagebox.showinfo("Sonuç", "Silhouette Score: {:.2f}".format(score))
        elif self.model_type == "dbscan":
            y_pred = self.model.fit_predict(X_test)
            score = calinski_harabasz_score(X_test, y_pred)
            messagebox.showinfo("Sonuç", "Calinski Score: {:.2f}".format(score))

        else:
            return None
        

def browse_file():
    filename = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")],
        title="Veri dosyasını seçin"
    )
    entry_file_path.delete(0, tk.END)
    entry_file_path.insert(tk.END, filename)

    if filename:
        data = pd.read_csv(filename)
        cols = list(data.columns)
        listbox_independent_cols.delete(0, tk.END)
        for col in cols:
            listbox_independent_cols.insert(tk.END, col)

def run_clustering():
    model_type = combobox_model_type.get()
    filename = entry_file_path.get()
    test_size = float(entry_test_size.get())
    random_state = int(entry_random_state.get())

    if not filename:
        messagebox.showerror("Hata", "Lütfen bir veri dosyası seçin!")
        return
    independent_cols = [listbox_independent_cols.get(index) for index in listbox_independent_cols.curselection()]

    if not independent_cols:
        messagebox.showerror("Hata", "Lütfen bağımsız değişkenleri seçin!")
        return

    n_cluster_value = None
    max_iter_value = None
    distance_value = None
    sample_value = None
    metric_value = None
    if model_type == "kmeans":
        try:
            n_cluster_value = int(entry_n_cluster.get())
            max_iter_value = int(entry_max_iter.get())
        except ValueError:
            messagebox("Hata","Girilmesi gereken değerleri giriniz.")
            return
    elif model_type == "agc":
        try:
            n_cluster_value = int(entry_n_cluster.get())
        except ValueError:
            messagebox("Hata","Girilmesi gereken değerleri giriniz.")
            return
    elif model_type == "dbscan":
        try:
            distance_value = float(entry_distance.get())
            sample_value = int(entry_sample.get())
            metric_value = str(entry_metric.get())  
        except ValueError:
            messagebox("Hata","Girilmesi gereken değerleri giriniz.")
            return

    clustering_models = ClusteringModels(model_type, n_cluster=n_cluster_value, max_iter = max_iter_value, distance = distance_value, sample = sample_value, metric = metric_value)
    X,cols = clustering_models.load_data(filename, independent_cols)
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
    clustering_models.clustering_model()
    clustering_models.train_model(X_train)
    clustering_models.evaluate_model(X_test)
    

# Arayüz oluşturma

root = tk.Tk()
root.title("Kümeleme Modelleri")

label_file_path = tk.Label(root, text="Veri Dosyası:")
label_file_path.grid(row=0, column=0, padx=10, pady=10)

entry_file_path = tk.Entry(root, width=50)
entry_file_path.grid(row=0, column=1, padx=10, pady=10)

button_browse = tk.Button(root, text="Gözat", command=browse_file)
button_browse.grid(row=0, column=2, padx=10, pady=10)

label_independent_cols = tk.Label(root, text="Bağımsız Değişkenler:")
label_independent_cols.grid(row=1, column=0, padx=10, pady=10)

listbox_independent_cols = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50)
listbox_independent_cols.grid(row=1, column=1, padx=10, pady=10)

label_model_type = tk.Label(root, text="Model Türü:")
label_model_type.grid(row=2, column=0, padx=10, pady=10)

combobox_model_type = ttk.Combobox(root, values=["kmeans", "agc", "dbscan"], state="readonly")
combobox_model_type.grid(row=2, column=1, padx=10, pady=10)

label_n_cluster = tk.Label(root, text="Küme Sayısı(örn:3): \n Kullanılacak Algortima: Kmeans / AGC")
label_n_cluster.grid(row=3, column=0, padx=10, pady=10)

entry_n_cluster = tk.Entry(root)
entry_n_cluster.grid(row=3, column=1, padx=10, pady=10)

label_max_iter = tk.Label(root, text="İterasyon Sayısı(örn:10): \n Kullanılacak Algoritma: KMeans")
label_max_iter.grid(row=4, column=0, padx=10, pady=10)

entry_max_iter = tk.Entry(root)
entry_max_iter.grid(row=4, column=1, padx=10, pady=10)

label_distance = tk.Label(root, text="Mesafe(örn:0.5):\n Kullanılacak Algoritma: DBSCAN")
label_distance.grid(row=5, column=0, padx=10, pady=10)

entry_distance = tk.Entry(root)
entry_distance.grid(row=5, column=1, padx=10, pady=10)

label_sample = tk.Label(root, text="Minimum Örnek(örn:5):\n Kullanılacak Algoritma: DBSCAN")
label_sample.grid(row=6, column=0, padx=10, pady=10)

entry_sample = tk.Entry(root)
entry_sample.grid(row=6, column=1, padx=10, pady=10)

label_metric = tk.Label(root, text="Metrik(örn:euclidean): \nKullanılacak Algoritma: DBSCAN")
label_metric.grid(row=7, column=0, padx=10, pady=10)

entry_metric = tk.Entry(root)
entry_metric.grid(row=7, column=1, padx=10, pady=10)

label_test_size = tk.Label(root, text="Test Boyutu:")
label_test_size.grid(row=9, column=0, padx=10, pady=10)

entry_test_size = tk.Entry(root)
entry_test_size.grid(row=9, column=1, padx=10, pady=10)

label_random_state = tk.Label(root, text="Rastgele Durum:")
label_random_state.grid(row=10, column=0, padx=10, pady=10)

entry_random_state = tk.Entry(root)
entry_random_state.grid(row=10, column=1, padx=10, pady=10)

button_run = tk.Button(root, text="Kümele", command=run_clustering)
button_run.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()


