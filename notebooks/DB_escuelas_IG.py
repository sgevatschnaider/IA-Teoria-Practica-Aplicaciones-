# -*- coding: utf-8 -*-

from tkinter import Tk, messagebox, Menu, Frame, Button, Entry, Label, OptionMenu, StringVar, DoubleVar, END
import sqlite3 as sq3

'''
****************************************************************
*                                                              *
*                    PARTE FUNCIONAL (LÓGICA)                  *
*                                                              *
****************************************************************
'''

#---------------------------------------------------------------
#                   FUNCIONES DEL MENÚ
#---------------------------------------------------------------

def conectar():
    """Establece la conexión global con la base de datos."""
    global con, cur
    try:
        con = sq3.connect('mi_db.db')
        cur = con.cursor()
        messagebox.showinfo("Conexión Exitosa", "Se ha conectado correctamente a la base de datos.")
    except sq3.Error as e:
        messagebox.showerror("Error de Conexión", f"No se pudo conectar a la base de datos: {e}")

def salir():
    """Cierra la conexión a la BBDD y termina la aplicación."""
    resp = messagebox.askquestion("Confirmar Salida", "¿Está seguro de que desea salir?")
    if resp == "yes":
        try:
            # Es una buena práctica cerrar la conexión si existe.
            if 'con' in globals() and con:
                con.close()
        except sq3.Error as e:
            messagebox.showwarning("Advertencia", f"Error al cerrar la base de datos: {e}")
        
        raiz.destroy()

def limpiar_campos():
    """Limpia todos los campos de entrada del formulario."""
    legajo.set("")
    alumno.set("")
    email.set("")
    calificacion.set(0.0) # Un valor por defecto para DoubleVar
    escuela.set("Seleccione una escuela") # Texto guía
    localidad.set("")
    provincia.set("")
    legajo_input.config(state='normal') # Habilita el campo legajo para nuevas búsquedas/creaciones

def mostrar_acercade():
    """Muestra una ventana con información sobre la aplicación."""
    messagebox.showinfo("Acerca de", "Sistema de Gestión de Alumnos\n\nDesarrollado en Python con Tkinter y SQLite.")

#---------------------------------------------------------------
#               FUNCIONES AUXILIARES Y DE LISTADO
#---------------------------------------------------------------

def buscar_escuelas(obtener_detalles):
    """
    Busca escuelas en la BBDD.
    Si obtener_detalles es False, devuelve una lista de nombres.
    Si obtener_detalles es True, devuelve el id, localidad y provincia de la escuela seleccionada.
    """
    # Esta función usa una conexión local para no interferir con la conexión global.
    # Es más seguro, ya que se abre y se cierra en la misma operación.
    try:
        with sq3.connect('mi_db.db') as con_local:
            cur_local = con_local.cursor()
            if obtener_detalles:
                cur_local.execute('SELECT _id, localidad, provincia FROM escuelas WHERE nombre =?', (escuela.get(),))
            else: 
                cur_local.execute('SELECT nombre FROM escuelas ORDER BY nombre ASC')
            
            resultado = cur_local.fetchall()
            retorno = []
            for item in resultado:
                if obtener_detalles:            
                    provincia.set(item[2])
                    localidad.set(item[1])
                    retorno.append(item[0]) # Devuelve el ID de la escuela
                else:
                    retorno.append(item[0]) # Devuelve el nombre de la escuela
            return retorno
    except sq3.Error as e:
        messagebox.showerror("Error de Consulta", f"No se pudieron cargar las escuelas: {e}")
        return []

def listar_alumnos():
    """Crea una nueva ventana para mostrar un listado completo de todos los alumnos."""
    
    # Se define una clase interna para construir la tabla de forma dinámica.
    class Tabla:
        def __init__(self, frame_tabla, data):
            cant_filas = len(data)
            cant_cols = len(data[0]) if cant_filas > 0 else 0
            
            # Encabezados de la tabla
            nombre_cols = ['Legajo', 'Alumno', 'Nota', 'Email', 'Escuela', 'Localidad', 'Provincia']
            
            for i, col_name in enumerate(nombre_cols):
                self.e = Entry(frame_tabla, width=22, bg='black', fg='white', font=('Arial', 10, 'bold'), justify='center')
                self.e.grid(row=0, column=i)
                self.e.insert(END, col_name)
                self.e.config(state='readonly')

            # Contenido de la tabla
            for fila in range(cant_filas):
                for col in range(cant_cols):
                    self.e = Entry(frame_tabla, width=22, font=('Arial', 10))
                    self.e.grid(row=fila + 1, column=col)
                    # Si el valor es None (nota nula), se muestra un string vacío.
                    valor = data[fila][col] if data[fila][col] is not None else ""
                    self.e.insert(END, valor)
                    self.e.config(state='readonly')

    # Configuración de la ventana secundaria
    raiz_lista = Tk()
    raiz_lista.title('Listado Completo de Alumnos')
    raiz_lista.config(bg='black')

    frame_principal = Frame(raiz_lista)    
    frame_principal.pack(fill='both', expand=True, padx=5, pady=5)
    
    try:
        # Usa una conexión local para esta operación aislada
        with sq3.connect('mi_db.db') as con_local:
            cur_local = con_local.cursor()
            query = '''
                    SELECT al.legajo, al.nombre, al.nota, al.email, 
                           es.nombre, es.localidad, es.provincia
                    FROM alumnos al INNER JOIN escuelas es
                    ON al.id_escuela = es._id
                    ORDER BY es.provincia, es.nombre, al.nombre
                    '''
            cur_local.execute(query)
            resultado = cur_local.fetchall()
        
        if resultado:
            tabla = Tabla(frame_principal, resultado)
        else:
            Label(frame_principal, text="No hay alumnos para mostrar.", fg="white", bg="black").pack()

    except sq3.Error as e:
        raiz_lista.destroy()
        messagebox.showerror("Error de Base de Datos", f"No se pudo realizar la consulta: {e}")
        return

    # Botón para cerrar la ventana de listado
    Button(raiz_lista, text="CERRAR VENTANA", command=raiz_lista.destroy, bg='red', fg='white', font=('Arial', 10, 'bold')).pack(fill='x', padx=5, pady=5)
    
    raiz_lista.mainloop()


#---------------------------------------------------------------
#         FUNCIONES CRUD (Create, Read, Update, Delete)
#---------------------------------------------------------------

def crear():
    """Inserta un nuevo registro de alumno en la base de datos."""
    if not 'con' in globals():
        messagebox.showerror("ERROR", "Primero debe conectar a la base de datos desde el menú BBDD.")
        return
    
    resp = messagebox.askquestion("Confirmar Creación", "¿Desea agregar este nuevo alumno?")
    if resp == 'yes':
        try:
            id_escuela_lista = buscar_escuelas(True)
            if not id_escuela_lista:
                messagebox.showerror("Error de Validación", "Debe seleccionar una escuela válida.")
                return
            
            id_escuela = int(id_escuela_lista[0])
            datos = (id_escuela, legajo.get(), alumno.get(), calificacion.get(), email.get())
            
            cur.execute("INSERT INTO alumnos (id_escuela, legajo, nombre, nota, email) VALUES (?,?,?,?,?)", datos)
            con.commit()
            messagebox.showinfo("Operación Exitosa", "Registro de alumno creado correctamente.")
            limpiar_campos()
        except sq3.IntegrityError:
            messagebox.showerror("Error de Duplicado", f"El legajo N° {legajo.get()} ya existe en la base de datos.")
        except Exception as e:
            messagebox.showerror("Error en Creación", f"No se pudo crear el registro: {e}")

def buscar_legajo():
    """Busca un alumno por su número de legajo y muestra sus datos."""
    if not 'con' in globals():
        messagebox.showerror("ERROR", "Primero debe conectar a la base de datos.")
        return
        
    try:
        limpiar_campos()
        query_buscar ='''
            SELECT al.legajo, al.nombre, al.nota, al.email,
                   es.nombre, es.localidad, es.provincia 
            FROM alumnos al INNER JOIN escuelas es
            ON al.id_escuela = es._id WHERE al.legajo = ?'''
        
        cur.execute(query_buscar, (legajo.get(),))
        resultado = cur.fetchone() # Usamos fetchone() porque legajo es UNIQUE
        
        if not resultado:
            messagebox.showerror("No Encontrado", f"El N° de legajo {legajo.get()} no existe.")
            legajo.set("")
        else:
            legajo.set(resultado[0])
            alumno.set(resultado[1])
            nota = resultado[2] if resultado[2] is not None else 0.0
            calificacion.set(nota)
            email.set(resultado[3])
            escuela.set(resultado[4])
            localidad.set(resultado[5])
            provincia.set(resultado[6])
            legajo_input.config(state='disabled') # Deshabilita el legajo para evitar cambios al actualizar/borrar
    except Exception as e:
        messagebox.showerror("Error en Búsqueda", f"Ocurrió un error al buscar: {e}")

def actualizar():
    """Actualiza los datos del alumno que se está mostrando en pantalla."""
    if not 'con' in globals():
        messagebox.showerror("ERROR", "Primero debe conectar a la base de datos.")
        return
    
    resp = messagebox.askquestion("Confirmar Actualización", "¿Desea guardar los cambios en este registro?")
    if resp == 'yes':
        try:
            id_escuela_lista = buscar_escuelas(True)
            if not id_escuela_lista:
                messagebox.showerror("Error de Validación", "Debe seleccionar una escuela válida.")
                return

            id_escuela = int(id_escuela_lista[0])
            datos = (id_escuela, alumno.get(), calificacion.get(), email.get(), legajo.get())
            
            query_update = "UPDATE alumnos SET id_escuela=?, nombre=?, nota=?, email=? WHERE legajo=?"
            cur.execute(query_update, datos)
            con.commit()
            
            messagebox.showinfo("Operación Exitosa", "Registro actualizado correctamente.")
            limpiar_campos()
        except Exception as e:
            messagebox.showerror("Error en Actualización", f"No se pudo actualizar el registro: {e}")

def borrar():
    """Elimina el registro del alumno que se está mostrando."""
    if not 'con' in globals():
        messagebox.showerror("ERROR", "Primero debe conectar a la base de datos.")
        return
        
    resp = messagebox.askquestion("Confirmar Eliminación", f"¿Está seguro de que desea eliminar el alumno con legajo {legajo.get()}?")
    if resp == 'yes':
        try:
            cur.execute("DELETE FROM alumnos WHERE legajo = ?", (legajo.get(),))
            con.commit()
            messagebox.showinfo("Operación Exitosa", "Registro eliminado correctamente.")
            limpiar_campos()
        except Exception as e:
            messagebox.showerror("Error en Eliminación", f"No se pudo eliminar el registro: {e}")

'''
****************************************************************
*                                                              *
*          INTERFAZ GRÁFICA (VISTA) - CONSTRUCCIÓN             *
*                                                              *
****************************************************************
'''
# --- Raíz de la aplicación ---
raiz = Tk()
raiz.title('Sistema de Gestión de Alumnos')
raiz.geometry("480x450") # Tamaño inicial de la ventana

# --- Constantes de Estilo ---
FONDO_PRINCIPAL = '#ECECEC' # Un gris claro moderno
COLOR_LETRA = '#333333' # Gris oscuro para el texto
FONDO_BOTONERA = '#D0D0D0' # Un gris medio para la botonera
COLOR_FONDO_BOTON = '#0078D7' # Azul de Windows
COLOR_TEXTO_BOTON = 'white'
FONDO_PIE = 'black'
COLOR_TEXTO_PIE = 'white'

raiz.config(bg=FONDO_PRINCIPAL)

#---------------------------------------------------------------
#                        BARRA DE MENÚ
#---------------------------------------------------------------
barramenu = Menu(raiz)
raiz.config(menu=barramenu)

# Menú BBDD
bbddmenu = Menu(barramenu, tearoff=0)
bbddmenu.add_command(label='Conectar a la BBDD', command=conectar)
bbddmenu.add_separator()
bbddmenu.add_command(label='Listado de alumnos', command=listar_alumnos)
bbddmenu.add_separator()
bbddmenu.add_command(label='Salir', command=salir)

# Menú Limpiar
limpiarmenu = Menu(barramenu, tearoff=0)
limpiarmenu.add_command(label='Limpiar campos del formulario', command=limpiar_campos)

# Menú Ayuda
ayudamenu = Menu(barramenu, tearoff=0)
ayudamenu.add_command(label='Acerca de...', command=mostrar_acercade)

barramenu.add_cascade(label='BBDD', menu=bbddmenu)
barramenu.add_cascade(label='Limpiar', menu=limpiarmenu)
barramenu.add_cascade(label='Ayuda', menu=ayudamenu)

#---------------------------------------------------------------
#                 FRAME DE CAMPOS DE ENTRADA
#---------------------------------------------------------------
frame_campos = Frame(raiz)
frame_campos.config(bg=FONDO_PRINCIPAL)
frame_campos.pack(fill='both', expand=True, padx=10, pady=5)

# Variables de control para los widgets
legajo = StringVar()
alumno = StringVar()
email = StringVar()
calificacion = DoubleVar()
escuela = StringVar()
localidad = StringVar()
provincia = StringVar()

# --- Widgets de Entrada (Entry, OptionMenu) ---
legajo_input = Entry(frame_campos, textvariable=legajo)
legajo_input.grid(row=0, column=1, padx=10, pady=10)

alumno_input = Entry(frame_campos, textvariable=alumno, width=40)
alumno_input.grid(row=1, column=1, padx=10, pady=10)

email_input = Entry(frame_campos, textvariable=email, width=40)
email_input.grid(row=2, column=1, padx=10, pady=10)

calificacion_input = Entry(frame_campos, textvariable=calificacion)
calificacion_input.grid(row=3, column=1, padx=10, pady=10)

# Carga las escuelas al inicio. El try/except previene un crash si la BBDD no existe aún.
try:
    lista_escuelas = buscar_escuelas(False)
except sq3.OperationalError:
    messagebox.showwarning("Atención", "No se encontró la base de datos 'mi_db.db'.\n" \
                                      "Asegúrese de ejecutar primero el script que la crea.\n" \
                                      "Algunas funciones estarán deshabilitadas.")
    lista_escuelas = ["(Base de datos no encontrada)"]

escuela.set('Seleccione una escuela')
escuela_option = OptionMenu(frame_campos, escuela, *lista_escuelas)
escuela_option.config(width=30)
escuela_option.grid(row=4, column=1, padx=10, pady=10, sticky='w')

# Entradas de solo lectura para mostrar info de la escuela
localidad_input = Entry(frame_campos, textvariable=localidad, width=40, state='readonly')
localidad_input.grid(row=5, column=1, padx=10, pady=10)

provincia_input = Entry(frame_campos, textvariable=provincia, width=40, state='readonly')
provincia_input.grid(row=6, column=1, padx=10, pady=10)

# --- Etiquetas (Labels) ---
def configurar_label(texto, fila):
    """Función auxiliar para crear y posicionar las etiquetas."""
    label = Label(frame_campos, text=texto, bg=FONDO_PRINCIPAL, fg=COLOR_LETRA)
    label.grid(row=fila, column=0, sticky='e', padx=10, pady=10)

configurar_label('N° de Legajo:', 0)
configurar_label('Nombre del Alumno:', 1)
configurar_label('Email:', 2)
configurar_label('Calificación:', 3)
configurar_label('Escuela:', 4)
configurar_label('Localidad:', 5)
configurar_label('Provincia:', 6)

#---------------------------------------------------------------
#                 FRAME DE BOTONES (CRUD)
#---------------------------------------------------------------
frame_botones = Frame(raiz)
frame_botones.config(bg=FONDO_BOTONERA)
frame_botones.pack(fill='x', side='bottom', ipady=5)
frame_botones.grid_columnconfigure((0, 1, 2, 3), weight=1) # Centra los botones

def configurar_boton(texto, comando, columna):
    """Función auxiliar para crear y posicionar los botones del CRUD."""
    boton = Button(frame_botones, text=texto, command=comando, bg=COLOR_FONDO_BOTON, fg=COLOR_TEXTO_BOTON, width=10)
    boton.grid(row=0, column=columna, padx=5, pady=5, ipadx=5)

configurar_boton('Crear', crear, 0)
configurar_boton('Buscar', buscar_legajo, 1)
configurar_boton('Actualizar', actualizar, 2)
configurar_boton('Eliminar', borrar, 3)

#---------------------------------------------------------------
#                 FRAME DEL PIE DE PÁGINA
#---------------------------------------------------------------
frame_pie = Frame(raiz)
frame_pie.config(bg=FONDO_PIE)
frame_pie.pack(fill='x', side='bottom')

pie_label = Label(frame_pie, text="Sistema de Gestión de Alumnos © 2025", bg=FONDO_PIE, fg=COLOR_TEXTO_PIE)
pie_label.pack(pady=5)

# --- Bucle principal de la aplicación ---
raiz.mainloop()