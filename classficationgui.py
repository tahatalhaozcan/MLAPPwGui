
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox, filedialog

class ClassificationModels:
    def __init__(self, model_type, criterion=None, splitter=None, max_depth=None,
                 n_estimators=None, n_neighbors=None, metric=None, weights=None,
                 var_smoothing=None):
        self.model_type = model_type
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.var_smoothing = var_smoothing
        self.model = None

    def load_data(self, filename, independent_cols, dependent_col):
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
        y = data[dependent_col].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y, cols

    def class_model(self):
        if self.model_type == 'dtc':
            if self.criterion is not None and self.splitter is not None and self.max_depth is not None:
                self.model = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter, max_depth=self.max_depth)
            else:
                raise ValueError("DTC için kriter, splitter ve maksimum derinlik girmelisiniz")

        elif self.model_type == "rfc":
            if self.criterion is not None and self.n_estimators is not None and self.max_depth is not None:
                self.model = RandomForestClassifier(criterion=self.criterion, n_estimators=self.n_estimators, max_depth=self.max_depth)
            else:
                raise ValueError("RFC için kriter, ağaç sayısı ve maksimum derinlik girmelisiniz")

        elif self.model_type == "knn":
            if self.n_neighbors is not None and self.metric is not None and self.weights is not None:
                self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric, weights=self.weights)
            else:
                raise ValueError("KNN için komşu sayısı, metrik ve ağırlık değerlerini girmelisiniz")

        elif self.model_type == "gnb":
            if self.var_smoothing is not None:
                self.model = GaussianNB(var_smoothing=self.var_smoothing)
            else:
                raise ValueError("GNB için Smoothing değeri giriniz.")

        else:
            return None

    def train_model(self, X_train, y_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)


    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(y_test)
        print(y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))

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
            
def run_classification_models():
        model_type = var_model_type.get()
        filename = entry_file_path.get()
        test_size = float(entry_test_size.get())
        random_state = int(entry_random_state.get())

        if not filename:
            messagebox.showerror("Hata", "Lütfen bir veri dosyası seçin!")
            return

        independent_cols = [listbox_independent_cols.get(index) for index in listbox_independent_cols.curselection()]
        dependent_col = entry_dependent_col.get()

        if not independent_cols:
            messagebox.showerror("Hata", "Lütfen en az bir bağımsız değişken seçin!")
            return

        if dependent_col == "":
            messagebox.showerror("Hata", "Lütfen bir bağımlı değişken girin!")
            return

        criterion_value = None
        splitter_value = None
        max_depth_value = None
        n_estimators_value = None
        n_neighbors_value = None
        metric_value = None
        weights_value = None
        var_smoothing_value = None

        if model_type == "dtc":
            try:
                criterion_value = str(entry_criterion.get())
                splitter_value = str(entry_splitter.get())
                max_depth_value = int(entry_max_depth.get())
            except ValueError:
                messagebox.showerror("Hata", "Girmediğiniz Kriter, Splitter ve/veya Derinlik değeri giriniz.")
                return
        elif model_type == "rfc":
            try:
                n_estimators_value = int(entry_n_estimators.get())
                criterion_value = str(entry_criterion.get())
                max_depth_value = int(entry_max_depth.get())
            except ValueError:
                messagebox.showerror("Hata", "Girmediğiniz Kriter, Ağaç Sayısı ve/veya Derinlik değeri giriniz.")
                return
        elif model_type == "knn":
            try:
                n_neighbors_value = int(entry_n_neighbors.get())
                metric_value = str(entry_metric.get())
                weights_value = str(entry_weights.get())
            except ValueError:
                messagebox.showerror("Hata", "Girmediğiniz Komşu Sayısı, Metrik türü ve/veya Ağırlık Türü değeri giriniz.")
                return
        elif model_type == "gnb":
            try:
                var_smoothing_value = float(entry_var_smoothing.get())
            except ValueError:
                messagebox.showerror("Hata", "Girmediğiniz Var Smoothing değeri giriniz.")
                return

        model = ClassificationModels(model_type, criterion=criterion_value, splitter=splitter_value,
                                     max_depth=max_depth_value, n_estimators=n_estimators_value,
                                     n_neighbors=n_neighbors_value, metric=metric_value, weights=weights_value,
                                     var_smoothing=var_smoothing_value)
        X, y, cols = model.load_data(filename, independent_cols, dependent_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model.class_model()
        model.train_model(X_train, y_train)
        model.evaluate_model(X_test, y_test)
        y_pred = model.predict(X_test)
        messagebox.showinfo("Sonuçlar", f"Gerçek Değerler: {y_test}\nTahmin Edilen Değerler: {y_pred}\nAccuracy: {accuracy_score(y_test, y_pred)}")


# Kullanıcı arayüzü
root = tk.Tk()
root.title("Sınıflandırma Modelleri")

frame_file = tk.Frame(root)
frame_file.pack(pady=10)

label_file = tk.Label(frame_file, text="Veri Dosyası:")
label_file.pack(side=tk.LEFT)

entry_file_path = tk.Entry(frame_file, width=50)
entry_file_path.pack(side=tk.LEFT, padx=10)

button_browse = tk.Button(frame_file, text="Gözat", command=browse_file)
button_browse.pack(side=tk.LEFT)

frame_data_columns = tk.Frame(root)
frame_data_columns.pack(pady=10)

label_data_columns = tk.Label(frame_data_columns, text="Veri Sütunları \n Bağımsız değişkenleri seçiniz: ")
label_data_columns.pack(side=tk.LEFT)

scrollbar_data_columns = tk.Scrollbar(frame_data_columns)
scrollbar_data_columns.pack(side=tk.RIGHT, fill=tk.Y)

listbox_independent_cols = tk.Listbox(frame_data_columns, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar_data_columns.set)
listbox_independent_cols.pack(side=tk.LEFT, fill=tk.BOTH)
scrollbar_data_columns.config(command=listbox_independent_cols.yview)

frame_dependent_col = tk.Frame(root)
frame_dependent_col.pack(pady=10)

label_dependent_col = tk.Label(frame_dependent_col, text="Bağımlı Değişken:")
label_dependent_col.pack(side=tk.LEFT)

entry_dependent_col = tk.Entry(frame_dependent_col, width=30)
entry_dependent_col.pack(side=tk.LEFT)

frame_options = tk.Frame(root)
frame_options.pack(pady=10)

label_model_type = tk.Label(frame_options, text="Model Türü:")
label_model_type.pack(side=tk.LEFT)

var_model_type = tk.StringVar()
var_model_type.set("dtc")

option_menu_model_type = tk.OptionMenu(frame_options, var_model_type, "dtc", "rfc", "knn", "gnb")
option_menu_model_type.pack(side=tk.LEFT)

frame_params = tk.Frame(root)
frame_params.pack(pady=10)

label_criterion = tk.Label(frame_params, text="Kriter Değeri(gini/entropy): \nKullanılabilecek Algoritmalar:DTC - RFC")
label_criterion.pack(side=tk.LEFT)

entry_criterion = tk.Entry(frame_params, width=10)
entry_criterion.pack(side=tk.LEFT, padx=10)

label_splitter = tk.Label(frame_params, text="Splitter Değeri(best/random):  \nKullanılabilecek Algoritmalar: DTC")
label_splitter.pack(side=tk.LEFT)

entry_splitter = tk.Entry(frame_params, width=10)
entry_splitter.pack(side=tk.LEFT, padx=10)

label_max_depth = tk.Label(frame_params, text="Derinlik Değeri(örn:3): \nKullanılabilecek Algoritmalar: DTC-RFC")
label_max_depth.pack(side=tk.LEFT)

entry_max_depth = tk.Entry(frame_params, width=10)
entry_max_depth.pack(side=tk.LEFT, padx=10)

frame_params = tk.Frame(root)
frame_params.pack(pady=10)

label_n_estimators = tk.Label(frame_params, text="Ağaç Sayısı(örn:3):  \nKullanılabilecek Algoritmalar: RFC")
label_n_estimators.pack(side=tk.LEFT)

entry_n_estimators = tk.Entry(frame_params, width=10)
entry_n_estimators.pack(side=tk.LEFT, padx=10)

label_n_neighbors = tk.Label(frame_params, text="Komşu Sayısı(örn:3): \nKullanılabilecek Algoritmalar: KNN")
label_n_neighbors.pack(side=tk.LEFT)

entry_n_neighbors = tk.Entry(frame_params, width=10)
entry_n_neighbors.pack(side=tk.LEFT, padx=10)

label_metric = tk.Label(frame_params, text="Metrik Değeri(örn: minkowkski): \nKullanılabilecek Algoritmalar: KNN")
label_metric.pack(side=tk.LEFT)

entry_metric = tk.Entry(frame_params, width=10)
entry_metric.pack(side=tk.LEFT, padx=10)

label_weights = tk.Label(frame_params, text="Ağırlık Türü(uniform/distance):  \nKullanılabilecek Algoritmalar: KNN")
label_weights.pack(side=tk.LEFT)

entry_weights = tk.Entry(frame_params, width=10)
entry_weights.pack(side=tk.LEFT, padx=10)

frame_params = tk.Frame(root)
frame_params.pack(pady=10)

label_var_smoothing = tk.Label(frame_params, text="Var Smoothing(örn:1e-9):  \nKullanılabilecek Algoritmalar: GNB")
label_var_smoothing.pack(side=tk.LEFT)

entry_var_smoothing = tk.Entry(frame_params, width=10)
entry_var_smoothing.pack(side=tk.LEFT, padx=10)

label_test_size = tk.Label(frame_params, text="Test Verisi Oranı(0-1 aralğında):")
label_test_size.pack(side=tk.LEFT)

entry_test_size = tk.Entry(frame_params, width=10)
entry_test_size.pack(side=tk.LEFT, padx=10)

label_random_state = tk.Label(frame_params, text="Random State(örn:42): ")
label_random_state.pack(side=tk.LEFT)

entry_random_state = tk.Entry(frame_params, width=10)
entry_random_state.pack(side=tk.LEFT)

button_run = tk.Button(root, text="Modeli Çalıştır", command=run_classification_models)
button_run.pack(pady=10)

root.mainloop()
