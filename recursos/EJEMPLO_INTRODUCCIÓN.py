import tkinter as tk

ventana = tk.Tk()                     # ventana raíz
ventana.title("Pantalla principal")   # título
ventana.geometry("400x300")           # ancho x alto (equivale a width/height)

# ----- widgets -----
lbl = tk.Label(ventana, text="¡Hola, Tkinter!")
lbl.pack(pady=20)                     # padding vertical

def saludar():
    lbl.config(text="Botón pulsado")

btn = tk.Button(ventana, text="Saludar", command=saludar)
btn.pack()

# ----- arranca la aplicación -----
ventana.mainloop()
