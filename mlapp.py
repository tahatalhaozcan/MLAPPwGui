from tkinter import Tk, Button
import os
import mlappgui1
import classficationgui
import clusteringgui

def mlappgui1_ac():
    os.system("python mlappgui1.py")

def classificationgui_ac():
    os.system("python classficationgui.py")
def clusteringgui_ac():
    os.system("python clusteringgui.py")

def ana_pencere():
    # Ana pencerenin oluşturulması
    root = Tk()
    root.title("MLAPP")

    # Program 1 butonu
    program1_btn = Button(root, text="Tahmin Modelleri", command=mlappgui1_ac,width=20,height=5)
    program1_btn.pack()

    # Program 2 butonu
    program2_btn = Button(root, text="Sınıflandırma Modelleri", command=classificationgui_ac,width=20,height=5)
    program2_btn.pack()
    
    program3_btn = Button(root, text="Kümeleme Modelleri", command=clusteringgui_ac,width=20,height=5)
    program3_btn.pack()

    # Ana pencerenin main döngüsünün başlatılması
    root.mainloop()

# Ana pencerenin çağrılması
if __name__ == "__main__":
    ana_pencere()