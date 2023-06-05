import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tkinter as tk
from tkinter import filedialog, messagebox


class PredictionModels:
    def __init__(self, model_type,alpha = None,l1_ratio = None):
        self.model_type = model_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
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

    def pred_model(self):
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == "lasso":
            if self.alpha is not None:
                self.model = Lasso(alpha=self.alpha)
            else:
                raise ValueError("Lasso Modeli için bir Alpha değeri giriniz.")
        elif self.model_type == "ridge":
           if self.alpha is not None:
               self.model = Ridge(alpha=self.alpha)
           else:
               raise ValueError("Ridge Modeli için bir Alpha değeri giriniz.")
        elif self.model_type == "elasticnet":
            if self.alpha is not None and self.l1_ratio is not None:
                self.model = ElasticNet(alpha = self.alpha,l1_ratio = self.l1_ratio)
            else:
                raise ValueError("Elasticnet için bir Alpha değeri ve bir L1 Ratio değeri girmelisiniz. ")
        else:
            return None

    def train_model(self, X_train, y_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def degerlendirme(self, X_test, y_test):
        y_pred = self.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(y_test)
        print(y_pred)
        print('R2 Skoru:', r2)


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


def run_prediction_model():
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

    alpha_value = None
    l1_value = None

    if model_type == "lasso" or model_type == "ridge":
        try:
            alpha_value = float(entry_alpha.get())
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli bir Alpha değeri giriniz.")
            return

    if model_type == "elasticnet":
        try:
            alpha_value = float(entry_alpha.get())
            l1_value = float(entry_l1.get())
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli bir Alpha ve L1 Ratio değeri giriniz.")
            return

    model = PredictionModels(model_type, alpha=alpha_value, l1_ratio=l1_value)  # alpha ve l1_ratio değerlerini geçir
    X, y, cols = model.load_data(filename, independent_cols, dependent_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model.pred_model()
    model.train_model(X_train, y_train)
    model.degerlendirme(X_test, y_test)
    y_pred = model.predict(X_test)
    messagebox.showinfo("Sonuçlar", f"Gerçek Değerler: {y_test}\nTahmin Edilen Değerler: {y_pred}\nR2 Skoru: {r2_score(y_test, y_pred)}")

# Kullanıcı arayüzü
root = tk.Tk()
root.title("Tahmin Modelleri")

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
var_model_type.set("linear")

option_menu_model_type = tk.OptionMenu(frame_options, var_model_type, "linear", "lasso", "ridge", "elasticnet")
option_menu_model_type.pack(side=tk.LEFT)

frame_params = tk.Frame(root)
frame_params.pack(pady=10)

label_alpha = tk.Label(frame_params, text="Alpha Değeri:  \nKullanılabilecek Algoritmalar: Lasso,Ridge,ElasticNet")
label_alpha.pack(side=tk.LEFT)

entry_alpha = tk.Entry(frame_params, width=10)
entry_alpha.pack(side=tk.LEFT, padx=10)

label_l1 = tk.Label(frame_params, text="L1 Değeri: \nKullanılabilecek Algoritmalar: ElasticNet" )
label_l1.pack(side=tk.LEFT)

entry_l1 = tk.Entry(frame_params, width=10)
entry_l1.pack(side=tk.LEFT, padx=10)

label_test_size = tk.Label(frame_params, text="Test Verisi Oranı:")
label_test_size.pack(side=tk.LEFT)

entry_test_size = tk.Entry(frame_params, width=10)
entry_test_size.pack(side=tk.LEFT, padx=10)

label_random_state = tk.Label(frame_params, text="Random State: ")
label_random_state.pack(side=tk.LEFT)

entry_random_state = tk.Entry(frame_params, width=10)
entry_random_state.pack(side=tk.LEFT)

button_run = tk.Button(root, text="Modeli Çalıştır", command=run_prediction_model)
button_run.pack(pady=10)

root.mainloop()
