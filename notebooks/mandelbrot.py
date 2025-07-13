import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from decimal import Decimal, getcontext

# Configurar precisión decimal para zooms extremos
getcontext().prec = 50

class MandelbrotFractalExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Explorador Fractal de Mandelbrot - Zoom Avanzado")
        self.root.geometry("1000x750")
        
        # Parámetros configurables
        self.image_width = 700
        self.image_height = 700
        self.max_iterations = 150
        
        # Coordenadas con alta precisión para zoom extremo
        self.min_real = Decimal('-2.5')
        self.max_real = Decimal('1.5')
        self.min_imaginary = Decimal('-2.0')
        self.max_imaginary = Decimal('2.0')
        
        # Variables de estado
        self.is_calculating = False
        self.current_image = None
        self.zoom_level = 1.0
        self.zoom_history = []
        self.selection_start = None
        self.selection_rect = None
        
        # Puntos de interés fractal predefinidos
        self.interesting_points = {
            "Vista General": (Decimal('-2.5'), Decimal('1.5'), Decimal('-2.0'), Decimal('2.0')),
            "Seahorse Valley": (Decimal('-0.75'), Decimal('-0.73'), Decimal('0.1'), Decimal('0.12')),
            "Lightning": (Decimal('-1.775'), Decimal('-1.76'), Decimal('-0.01'), Decimal('0.005')),
            "Spiral": (Decimal('-0.16'), Decimal('0.16'), Decimal('1.025'), Decimal('1.045')),
            "Mini Mandelbrot": (Decimal('-0.7463'), Decimal('-0.7453'), Decimal('0.1102'), Decimal('0.1112')),
            "Elephant Valley": (Decimal('0.25'), Decimal('0.26'), Decimal('0.0'), Decimal('0.01')),
            "Dendrite": (Decimal('-0.12'), Decimal('-0.11'), Decimal('0.85'), Decimal('0.86')),
            "Double Spiral": (Decimal('-0.8'), Decimal('-0.7'), Decimal('0.156'), Decimal('0.256'))
        }
        
        self.setup_ui()
        self.generate_mandelbrot()
    
    def setup_ui(self):
        """Configura la interfaz de usuario avanzada"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Panel de controles superior
        controls_frame = ttk.LabelFrame(main_frame, text="Controles de Exploración", padding="5")
        controls_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Fila 1 de controles
        control_row1 = ttk.Frame(controls_frame)
        control_row1.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(control_row1, text="Iteraciones:").grid(row=0, column=0, padx=(0, 5))
        self.iterations_var = tk.StringVar(value=str(self.max_iterations))
        iterations_spinbox = ttk.Spinbox(control_row1, from_=50, to=1000, width=8, 
                                       textvariable=self.iterations_var)
        iterations_spinbox.grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(control_row1, text="🔄 Generar", 
                  command=self.generate_mandelbrot).grid(row=0, column=2, padx=5)
        ttk.Button(control_row1, text="🏠 Reset", 
                  command=self.reset_view).grid(row=0, column=3, padx=5)
        ttk.Button(control_row1, text="⬅️ Atrás", 
                  command=self.zoom_back).grid(row=0, column=4, padx=5)
        ttk.Button(control_row1, text="💾 Guardar", 
                  command=self.save_image).grid(row=0, column=5, padx=5)
        
        # Fila 2 - Puntos de interés
        control_row2 = ttk.Frame(controls_frame)
        control_row2.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(control_row2, text="Puntos de Interés:").grid(row=0, column=0, padx=(0, 5))
        self.location_var = tk.StringVar(value="Vista General")
        location_combo = ttk.Combobox(control_row2, textvariable=self.location_var, 
                                    values=list(self.interesting_points.keys()),
                                    width=15, state="readonly")
        location_combo.grid(row=0, column=1, padx=(0, 10))
        location_combo.bind('<<ComboboxSelected>>', self.goto_location)
        
        # Información de zoom
        self.zoom_info = ttk.Label(control_row2, text="Zoom: 1.0x")
        self.zoom_info.grid(row=0, column=2, padx=10)
        
        # Controles de zoom
        ttk.Button(control_row2, text="🔍+ Zoom In", 
                  command=self.zoom_in_center).grid(row=0, column=3, padx=5)
        ttk.Button(control_row2, text="🔍- Zoom Out", 
                  command=self.zoom_out_center).grid(row=0, column=4, padx=5)
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(controls_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Frame principal del canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas para mostrar el fractal
        self.canvas = tk.Canvas(canvas_frame, width=self.image_width, height=self.image_height, 
                               bg='black', cursor='crosshair')
        self.canvas.grid(row=0, column=0)
        
        # Panel de información lateral
        info_frame = ttk.LabelFrame(main_frame, text="Información", padding="5")
        info_frame.grid(row=1, column=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Coordenadas actuales
        self.coord_label = ttk.Label(info_frame, text="Coordenadas:\nReal: -2.5 a 1.5\nImag: -2.0 a 2.0", 
                                   justify=tk.LEFT, font=('Courier', 9))
        self.coord_label.grid(row=0, column=0, sticky=(tk.W, tk.N), pady=5)
        
        # Información del punto bajo el cursor
        self.cursor_info = ttk.Label(info_frame, text="Punto:\nReal: \nImag: \nIteraciones: ", 
                                   justify=tk.LEFT, font=('Courier', 9))
        self.cursor_info.grid(row=1, column=0, sticky=(tk.W, tk.N), pady=5)
        
        # Tips de uso
        tips_text = """💡 CONTROLES:
        
🖱️ Clic y arrastra: Seleccionar zona
🖱️ Doble clic: Zoom in en punto
🖱️ Clic derecho: Zoom out
⌨️ Teclas +/-: Zoom centro
⌨️ Flechas: Mover vista
⌨️ R: Reset vista
⌨️ S: Guardar imagen

🔍 ZOOM TIPS:
• Usa puntos de interés
• Busca patrones repetitivos
• Explora bordes del conjunto
• Aumenta iteraciones para
  más detalle en zooms altos"""
        
        tips_label = ttk.Label(info_frame, text=tips_text, justify=tk.LEFT, 
                             font=('Arial', 8), foreground='blue')
        tips_label.grid(row=2, column=0, sticky=(tk.W, tk.N), pady=10)
        
        # Eventos del mouse y teclado
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_click)  # Clic derecho
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # Eventos de teclado
        self.root.bind("<Key>", self.on_key_press)
        self.root.focus_set()
        
        # Configurar redimensionamiento
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        controls_frame.columnconfigure(0, weight=1)
    
    def mandelbrot_calculation_optimized(self, real, imaginary, max_iter):
        """Cálculo optimizado del conjunto de Mandelbrot con escape suave"""
        z = complex(0, 0)
        c = complex(float(real), float(imaginary))
        
        for iteration in range(max_iter):
            if abs(z) > 2:
                # Escape suave para mejor coloración
                return iteration + 1 - np.log2(np.log2(abs(z)))
            z = z * z + c
        return max_iter
    
    def get_advanced_color(self, iterations):
        """Esquema de colores avanzado para mejor visualización fractal"""
        if iterations >= self.max_iterations:
            return (0, 0, 0)  # Negro para puntos en el conjunto
        
        # Normalizar iteraciones
        t = iterations / self.max_iterations
        
        # Esquema de colores con múltiples bandas
        if t < 0.16:
            # Azul profundo a azul brillante
            ratio = t / 0.16
            r = int(ratio * 100)
            g = int(ratio * 150)
            b = int(150 + ratio * 105)
        elif t < 0.33:
            # Azul a cyan
            ratio = (t - 0.16) / 0.17
            r = int(100 + ratio * 50)
            g = int(150 + ratio * 105)
            b = 255
        elif t < 0.5:
            # Cyan a verde
            ratio = (t - 0.33) / 0.17
            r = int(150 - ratio * 150)
            g = 255
            b = int(255 - ratio * 255)
        elif t < 0.66:
            # Verde a amarillo
            ratio = (t - 0.5) / 0.16
            r = int(ratio * 255)
            g = 255
            b = 0
        elif t < 0.83:
            # Amarillo a naranja
            ratio = (t - 0.66) / 0.17
            r = 255
            g = int(255 - ratio * 100)
            b = int(ratio * 100)
        else:
            # Naranja a rojo
            ratio = (t - 0.83) / 0.17
            r = 255
            g = int(155 - ratio * 155)
            b = int(100 - ratio * 100)
        
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    
    def generate_mandelbrot_array(self):
        """Genera el array del conjunto con alta precisión"""
        # Convertir a float para numpy (perdemos algo de precisión pero ganamos velocidad)
        min_real = float(self.min_real)
        max_real = float(self.max_real)
        min_imag = float(self.min_imaginary)
        max_imag = float(self.max_imaginary)
        
        # Crear arrays de coordenadas
        x = np.linspace(min_real, max_real, self.image_width)
        y = np.linspace(min_imag, max_imag, self.image_height)
        
        # Matriz de resultados
        iterations_array = np.zeros((self.image_height, self.image_width))
        
        # Calcular fila por fila para mostrar progreso
        for row in range(self.image_height):
            if not self.is_calculating:  # Permitir cancelación
                break
                
            for col in range(self.image_width):
                real = x[col]
                imaginary = y[row]
                iterations_array[row, col] = self.mandelbrot_calculation_optimized(
                    real, imaginary, self.max_iterations)
            
            # Actualizar progreso
            progress = ((row + 1) / self.image_height) * 100
            self.root.after(0, lambda p=progress: self.progress_var.set(p))
        
        return iterations_array
    
    def create_image_from_array(self, iterations_array):
        """Crea una imagen PIL a partir del array de iteraciones"""
        height, width = iterations_array.shape
        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                color = self.get_advanced_color(iterations_array[y, x])
                image_array[y, x] = color
        
        return Image.fromarray(image_array)
    
    def generate_mandelbrot(self):
        """Genera el conjunto de Mandelbrot en un hilo separado"""
        if self.is_calculating:
            return
        
        self.is_calculating = True
        self.max_iterations = int(self.iterations_var.get())
        
        # Actualizar información de coordenadas
        self.update_coordinate_display()
        
        def calculate():
            try:
                iterations_array = self.generate_mandelbrot_array()
                
                if self.is_calculating:  # Verificar si no fue cancelado
                    pil_image = self.create_image_from_array(iterations_array)
                    self.current_image = pil_image
                    self.photo = ImageTk.PhotoImage(pil_image)
                    self.root.after(0, self.update_canvas)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error generando el fractal: {str(e)}")
            finally:
                self.is_calculating = False
                self.root.after(0, lambda: self.progress_var.set(0))
        
        thread = threading.Thread(target=calculate, daemon=True)
        thread.start()
    
    def update_canvas(self):
        """Actualiza el canvas con la nueva imagen"""
        self.canvas.delete("all")
        self.canvas.create_image(self.image_width//2, self.image_height//2, 
                               image=self.photo, anchor=tk.CENTER)
    
    def pixel_to_complex(self, x, y):
        """Convierte coordenadas de pixel a coordenadas complejas con alta precisión"""
        real_range = self.max_real - self.min_real
        imag_range = self.max_imaginary - self.min_imaginary
        
        real = self.min_real + Decimal(x) * real_range / Decimal(self.image_width)
        imaginary = self.min_imaginary + Decimal(y) * imag_range / Decimal(self.image_height)
        
        return real, imaginary
    
    def save_zoom_state(self):
        """Guarda el estado actual del zoom"""
        state = (self.min_real, self.max_real, self.min_imaginary, self.max_imaginary, self.zoom_level)
        self.zoom_history.append(state)
        
        # Limitar historial a 20 estados
        if len(self.zoom_history) > 20:
            self.zoom_history.pop(0)
    
    def on_left_click(self, event):
        """Maneja el clic izquierdo"""
        self.selection_start = (event.x, event.y)
    
    def on_drag(self, event):
        """Maneja el arrastre para selección de zoom"""
        if self.selection_start and not self.is_calculating:
            if self.selection_rect:
                self.canvas.delete(self.selection_rect)
            
            # Crear rectángulo de selección proporcional
            start_x, start_y = self.selection_start
            width = abs(event.x - start_x)
            height = abs(event.y - start_y)
            
            # Mantener proporción cuadrada
            size = max(width, height)
            
            if event.x >= start_x:
                end_x = start_x + size
            else:
                end_x = start_x - size
            
            if event.y >= start_y:
                end_y = start_y + size
            else:
                end_y = start_y - size
            
            self.selection_rect = self.canvas.create_rectangle(
                start_x, start_y, end_x, end_y,
                outline='yellow', width=2, dash=(5, 5)
            )
    
    def on_left_release(self, event):
        """Maneja la liberación del clic izquierdo para zoom"""
        if self.selection_rect and self.selection_start:
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
            
            start_x, start_y = self.selection_start
            
            # Verificar si hay una selección válida
            if abs(event.x - start_x) > 10 and abs(event.y - start_y) > 10:
                self.zoom_to_selection(start_x, start_y, event.x, event.y)
        
        self.selection_start = None
    
    def on_double_click(self, event):
        """Zoom in en el punto del doble clic"""
        self.save_zoom_state()
        
        # Calcular nueva región centrada en el punto
        center_real, center_imag = self.pixel_to_complex(event.x, event.y)
        
        # Reducir región al 25%
        current_width = self.max_real - self.min_real
        current_height = self.max_imaginary - self.min_imaginary
        new_width = current_width / 4
        new_height = current_height / 4
        
        self.min_real = center_real - new_width / 2
        self.max_real = center_real + new_width / 2
        self.min_imaginary = center_imag - new_height / 2
        self.max_imaginary = center_imag + new_height / 2
        
        self.zoom_level *= 4
        self.update_zoom_display()
        self.generate_mandelbrot()
    
    def on_right_click(self, event):
        """Zoom out en el punto del clic derecho"""
        self.zoom_out_point(event.x, event.y)
    
    def on_mouse_move(self, event):
        """Actualiza información del punto bajo el cursor"""
        if hasattr(self, 'photo'):
            real, imag = self.pixel_to_complex(event.x, event.y)
            iterations = self.mandelbrot_calculation_optimized(real, imag, min(100, self.max_iterations))
            
            self.cursor_info.config(text=f"Punto:\nReal: {float(real):.8f}\nImag: {float(imag):.8f}\nIter: {iterations:.1f}")
    
    def on_key_press(self, event):
        """Maneja las teclas de acceso rápido"""
        key = event.keysym.lower()
        
        if key == 'plus' or key == 'equal':
            self.zoom_in_center()
        elif key == 'minus':
            self.zoom_out_center()
        elif key == 'r':
            self.reset_view()
        elif key == 's':
            self.save_image()
        elif key in ['up', 'down', 'left', 'right']:
            self.move_view(key)
    
    def zoom_to_selection(self, x1, y1, x2, y2):
        """Hace zoom a la región seleccionada"""
        self.save_zoom_state()
        
        # Calcular nuevas coordenadas
        real1, imag1 = self.pixel_to_complex(min(x1, x2), min(y1, y2))
        real2, imag2 = self.pixel_to_complex(max(x1, x2), max(y1, y2))
        
        self.min_real, self.max_real = real1, real2
        self.min_imaginary, self.max_imaginary = imag1, imag2
        
        # Calcular nuevo zoom level
        old_width = self.max_real - self.min_real
        self.zoom_level *= float(4.0 / abs(x2 - x1) * self.image_width)  # Aproximación
        
        self.update_zoom_display()
        self.generate_mandelbrot()
    
    def zoom_in_center(self):
        """Zoom in en el centro"""
        self.save_zoom_state()
        
        center_real = (self.min_real + self.max_real) / 2
        center_imag = (self.min_imaginary + self.max_imaginary) / 2
        
        width = (self.max_real - self.min_real) / 2
        height = (self.max_imaginary - self.min_imaginary) / 2
        
        self.min_real = center_real - width / 2
        self.max_real = center_real + width / 2
        self.min_imaginary = center_imag - height / 2
        self.max_imaginary = center_imag + height / 2
        
        self.zoom_level *= 2
        self.update_zoom_display()
        self.generate_mandelbrot()
    
    def zoom_out_center(self):
        """Zoom out en el centro"""
        self.save_zoom_state()
        
        center_real = (self.min_real + self.max_real) / 2
        center_imag = (self.min_imaginary + self.max_imaginary) / 2
        
        width = (self.max_real - self.min_real) * 2
        height = (self.max_imaginary - self.min_imaginary) * 2
        
        self.min_real = center_real - width / 2
        self.max_real = center_real + width / 2
        self.min_imaginary = center_imag - height / 2
        self.max_imaginary = center_imag + height / 2
        
        self.zoom_level /= 2
        self.update_zoom_display()
        self.generate_mandelbrot()
    
    def zoom_out_point(self, x, y):
        """Zoom out centrado en un punto específico"""
        self.save_zoom_state()
        
        center_real, center_imag = self.pixel_to_complex(x, y)
        
        width = (self.max_real - self.min_real) * 2
        height = (self.max_imaginary - self.min_imaginary) * 2
        
        self.min_real = center_real - width / 2
        self.max_real = center_real + width / 2
        self.min_imaginary = center_imag - height / 2
        self.max_imaginary = center_imag + height / 2
        
        self.zoom_level /= 2
        self.update_zoom_display()
        self.generate_mandelbrot()
    
    def move_view(self, direction):
        """Mueve la vista en la dirección especificada"""
        move_factor = 0.1  # 10% del rango actual
        
        width = self.max_real - self.min_real
        height = self.max_imaginary - self.min_imaginary
        
        if direction == 'up':
            delta = height * Decimal(move_factor)
            self.min_imaginary -= delta
            self.max_imaginary -= delta
        elif direction == 'down':
            delta = height * Decimal(move_factor)
            self.min_imaginary += delta
            self.max_imaginary += delta
        elif direction == 'left':
            delta = width * Decimal(move_factor)
            self.min_real -= delta
            self.max_real -= delta
        elif direction == 'right':
            delta = width * Decimal(move_factor)
            self.min_real += delta
            self.max_real += delta
        
        self.generate_mandelbrot()
    
    def zoom_back(self):
        """Vuelve al estado de zoom anterior"""
        if self.zoom_history:
            state = self.zoom_history.pop()
            self.min_real, self.max_real, self.min_imaginary, self.max_imaginary, self.zoom_level = state
            self.update_zoom_display()
            self.generate_mandelbrot()
    
    def goto_location(self, event=None):
        """Va a un punto de interés predefinido"""
        location = self.location_var.get()
        if location in self.interesting_points:
            self.save_zoom_state()
            
            coords = self.interesting_points[location]
            self.min_real, self.max_real, self.min_imaginary, self.max_imaginary = coords
            
            # Calcular zoom level aproximado
            width = float(self.max_real - self.min_real)
            self.zoom_level = 4.0 / width  # Aproximación basada en la vista inicial
            
            self.update_zoom_display()
            self.generate_mandelbrot()
    
    def reset_view(self):
        """Resetea la vista a los valores iniciales"""
        self.save_zoom_state()
        
        self.min_real = Decimal('-2.5')
        self.max_real = Decimal('1.5')
        self.min_imaginary = Decimal('-2.0')
        self.max_imaginary = Decimal('2.0')
        self.zoom_level = 1.0
        
        self.zoom_history.clear()
        self.location_var.set("Vista General")
        self.update_zoom_display()
        self.generate_mandelbrot()
    
    def update_zoom_display(self):
        """Actualiza la información del zoom en pantalla"""
        if self.zoom_level >= 1000000:
            zoom_text = f"Zoom: {self.zoom_level:.2e}x"
        elif self.zoom_level >= 1000:
            zoom_text = f"Zoom: {self.zoom_level:.0f}x"
        else:
            zoom_text = f"Zoom: {self.zoom_level:.1f}x"
        
        self.zoom_info.config(text=zoom_text)
    
    def update_coordinate_display(self):
        """Actualiza la visualización de coordenadas"""
        coord_text = f"Coordenadas:\nReal: {float(self.min_real):.8f} a {float(self.max_real):.8f}\nImag: {float(self.min_imaginary):.8f} a {float(self.max_imaginary):.8f}"
        self.coord_label.config(text=coord_text)
    
    def save_image(self):
        """Guarda la imagen actual con información del zoom"""
        if self.current_image:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")],
                initialname=f"mandelbrot_zoom_{self.zoom_level:.0f}x"
            )
            if filename:
                self.current_image.save(filename)
                messagebox.showinfo("Éxito", f"Imagen guardada como {filename}")
        else:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar")

if __name__ == "__main__":
    root = tk.Tk()
    app = MandelbrotFractalExplorer(root)
    root.mainloop()