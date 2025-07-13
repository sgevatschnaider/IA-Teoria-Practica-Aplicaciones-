import sqlite3 as sq3
import os # Importamos el módulo 'os' para interactuar con el sistema operativo

DB_FILE = 'mi_db.db'
# Elimina la base de datos anterior si existe para empezar de cero en cada ejecución.
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)
    print("Base de datos anterior eliminada. Creando una nueva.")

# Conectar a la base de datos (creará el archivo si no existe)
con = sq3.connect(DB_FILE)
cur = con.cursor()

print("Creando tablas...")

# --- Definición del Esquema de la Base de Datos (DDL) ---

# Tabla de escuelas
instruct1 = '''CREATE TABLE IF NOT EXISTS escuelas (
  _id INTEGER PRIMARY KEY AUTOINCREMENT,  
  nombre VARCHAR(50) NOT NULL,
  localidad VARCHAR(50),
  provincia VARCHAR(50),
  capacidad INTEGER)'''

# Tabla de alumnos
instruct2 = '''CREATE TABLE IF NOT EXISTS alumnos (
  _id INTEGER PRIMARY KEY AUTOINCREMENT,
  id_escuela INTEGER NOT NULL,
  legajo INTEGER UNIQUE,
  nombre VARCHAR(50) NOT NULL,
  nota DECIMAL(4,2),
  grado INTEGER,
  email VARCHAR(60),
  FOREIGN KEY (id_escuela) REFERENCES escuelas(_id))'''

cur.execute(instruct1)
cur.execute(instruct2)
print("Tablas 'escuelas' y 'alumnos' creadas con éxito.")

# --- Datos de Ejemplo para la Inserción ---

# NUEVOS DATOS: 8 escuelas diferentes
escuelas_lista = [
    ('Instituto Cervantes', 'Madrid', 'Madrid', 400),
    ('Colegio Pío IX', 'Almagro', 'Capital Federal', 350),
    ('Escuela Técnica Otto Krause', 'Monserrat', 'Capital Federal', 600),
    ('Liceo Militar General Paz', 'Córdoba', 'Córdoba', 200),
    ('Universidad Laboral', 'Gijón', 'Asturias', 550),
    ('Colegio de la Inmaculada', 'Lima', 'Lima', 300),
    ('Instituto Nacional', 'Santiago', 'Metropolitana', 700),
    ('Escuela Normal Superior', 'Manizales', 'Caldas', 250)
]

# NUEVOS DATOS: 18 alumnos diferentes, distribuidos en las nuevas escuelas.
# El id_escuela corresponde a la posición de la escuela en la lista de arriba (empezando en 1).
alumnos_lista = [
    (3, 2021, 'Ana López', 9.5, 5, 'ana.lopez@mail.com'),
    (1, 3015, 'Carlos García', 7.0, 3, 'cgarcia@mail.com'),
    (5, 4001, 'Beatriz Fernández', 8.8, 4, 'b.fernandez@mail.com'),
    (3, 2022, 'David Muñoz', None, 5, 'david.m@mail.com'), # Alumno con nota NULA
    (7, 5010, 'Elena Moreno', 10.0, 6, 'emoreno@mail.com'),
    (2, 1101, 'Francisco Jiménez', 6.2, 2, 'f.jimenez@mail.com'),
    (8, 6005, 'Gloria Ruiz', 7.9, 3, 'gloria.r@mail.com'),
    (4, 7020, 'Héctor Álvarez', 5.0, 1, 'h.alvarez@mail.com'),
    (1, 3016, 'Irene Romero', 8.1, 3, 'irene.romero@mail.com'),
    (6, 8001, 'Javier Alonso', 9.9, 5, 'jalonso@mail.com'),
    (7, 5012, 'Laura Navarro', None, 6, 'laura.n@mail.com'), # Alumna con nota NULA
    (5, 4008, 'Manuel Torres', 6.5, 4, 'mtorres@mail.com'),
    (2, 1105, 'Nuria Gutiérrez', 4.5, 2, 'n.gutierrez@mail.com'),
    (8, 6011, 'Oscar Prieto', 8.3, 3, 'oscar.p@mail.com'),
    (3, 2029, 'Patricia Castillo', 7.7, 5, ''), # Alumna con email VACÍO
    (6, 8003, 'Raquel Ramos', 9.1, 5, 'raquel.r@mail.com'),
    (7, 5018, 'Sergio Gil', 8.4, 6, 'sergio.gil@mail.com'),
    (4, 7025, 'Verónica Sanz', 6.9, 1, 'veronica.s@mail.com')
]

# --- Inserción de Datos ---
print("\nInsertando nuevos datos en las tablas...")
# Se especifican las columnas en la consulta INSERT para mayor claridad y seguridad.
cur.executemany('INSERT INTO escuelas (nombre, localidad, provincia, capacidad) VALUES (?,?,?,?)', escuelas_lista)
cur.executemany('INSERT INTO alumnos (id_escuela, legajo, nombre, nota, grado, email) VALUES (?,?,?,?,?,?)', alumnos_lista)
print(f"{len(escuelas_lista)} nuevas escuelas y {len(alumnos_lista)} nuevos alumnos insertados.")

# --- Consulta de Verificación ---
print("\n--- Verificación: Listado de Alumnos y sus Escuelas (con nuevos datos) ---")
query1 = '''SELECT
                alumnos.legajo,
                alumnos.nombre,
                alumnos.nota,
                escuelas.nombre AS nombre_escuela,
                escuelas.provincia
            FROM alumnos 
            INNER JOIN escuelas ON alumnos.id_escuela = escuelas._id
            ORDER BY escuelas.provincia, nombre_escuela, alumnos.nombre'''

for registro in cur.execute(query1):
    print(registro)

# Guardar los cambios (hacerlos permanentes) y cerrar la conexión
con.commit()
con.close()

print("\nProceso completado. La base de datos 'mi_db.db' ha sido creada y poblada con el nuevo set de datos.")