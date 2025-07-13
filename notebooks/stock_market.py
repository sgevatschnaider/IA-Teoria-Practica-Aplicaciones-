import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime, timedelta
import threading

class StockAnalyzer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Analizador de Acciones Profesional")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.current_ticker = None
        self.current_data = None
        
        # Crear interfaz
        self.create_widgets()
        
    def create_widgets(self) -> None:
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Frame de controles
        control_frame = ttk.LabelFrame(main_frame, text="Controles", padding=10)
        control_frame.pack(fill="x", pady=(0, 10))
        
        # Ticker y período
        ttk.Label(control_frame, text="Ticker:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ticker_entry = ttk.Entry(control_frame, width=12, font=("Arial", 10))
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5)
        self.ticker_entry.bind('<Return>', lambda e: self.fetch_and_analyze())
        
        ttk.Label(control_frame, text="Período:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.period_var = tk.StringVar(value="1y")
        period_combo = ttk.Combobox(control_frame, textvariable=self.period_var, width=8,
                                   values=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
        period_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # Botones
        ttk.Button(control_frame, text="Analizar", command=self.fetch_and_analyze).grid(
            row=0, column=4, padx=10, pady=5)
        ttk.Button(control_frame, text="Exportar Datos", command=self.export_data).grid(
            row=0, column=5, padx=5, pady=5)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, columnspan=6, sticky="ew", pady=5)
        
        # Notebook para pestañas
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Pestaña de información general
        self.info_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.info_frame, text="Información General")
        
        # Pestaña de análisis técnico
        self.tech_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tech_frame, text="Análisis Técnico")
        
        # Pestaña de métricas financieras
        self.metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_frame, text="Métricas Financieras")
        
        self.create_info_tab()
        self.create_tech_tab()
        self.create_metrics_tab()
        
    def create_info_tab(self) -> None:
        # Información básica
        info_label_frame = ttk.LabelFrame(self.info_frame, text="Información Básica", padding=10)
        info_label_frame.pack(fill="x", padx=10, pady=5)
        
        self.info_text = tk.Text(info_label_frame, height=12, wrap="word", font=("Consolas", 10))
        info_scrollbar = ttk.Scrollbar(info_label_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side="left", fill="both", expand=True)
        info_scrollbar.pack(side="right", fill="y")
        
        # Gráfico de precios
        chart_frame = ttk.LabelFrame(self.info_frame, text="Gráfico de Precios", padding=10)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.price_canvas_frame = ttk.Frame(chart_frame)
        self.price_canvas_frame.pack(fill="both", expand=True)
        
    def create_tech_tab(self) -> None:
        # Gráfico de análisis técnico
        tech_chart_frame = ttk.LabelFrame(self.tech_frame, text="Análisis Técnico", padding=10)
        tech_chart_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.tech_canvas_frame = ttk.Frame(tech_chart_frame)
        self.tech_canvas_frame.pack(fill="both", expand=True)
        
    def create_metrics_tab(self) -> None:
        # Métricas financieras
        metrics_label_frame = ttk.LabelFrame(self.metrics_frame, text="Métricas Clave", padding=10)
        metrics_label_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.metrics_text = tk.Text(metrics_label_frame, height=20, wrap="word", font=("Consolas", 10))
        metrics_scrollbar = ttk.Scrollbar(metrics_label_frame, orient="vertical", command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=metrics_scrollbar.set)
        
        self.metrics_text.pack(side="left", fill="both", expand=True)
        metrics_scrollbar.pack(side="right", fill="y")
        
    def fetch_and_analyze(self) -> None:
        """Ejecutar análisis en hilo separado para no bloquear UI"""
        threading.Thread(target=self._analyze_stock, daemon=True).start()
        
    def _analyze_stock(self) -> None:
        symbol = self.ticker_entry.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Aviso", "Ingresá un ticker válido.")
            return
            
        # Iniciar barra de progreso
        self.progress.start()
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Obtener datos
            info = ticker.info
            period = self.period_var.get()
            df = ticker.history(period=period)
            
            if df.empty:
                messagebox.showwarning("Aviso", "No hay datos históricos disponibles.")
                return
                
            self.current_ticker = symbol
            self.current_data = df
            
            # Actualizar UI en hilo principal
            self.root.after(0, self._update_ui, info, df)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo obtener datos: {e}")
        finally:
            self.progress.stop()
            
    def _update_ui(self, info: dict, df: pd.DataFrame) -> None:
        """Actualizar interfaz con datos obtenidos"""
        self.update_info_tab(info, df)
        self.update_tech_tab(df)
        self.update_metrics_tab(info, df)
        
    def update_info_tab(self, info: dict, df: pd.DataFrame) -> None:
        """Actualizar pestaña de información general"""
        # Información básica
        current_price = info.get('currentPrice', info.get('regularMarketPrice', df['Close'].iloc[-1]))
        prev_close = info.get('previousClose', df['Close'].iloc[-2] if len(df) > 1 else current_price)
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
        
        lines = [
            f"{'='*50}",
            f"INFORMACIÓN GENERAL - {self.current_ticker}",
            f"{'='*50}",
            f"Nombre: {info.get('shortName', info.get('longName', 'N/A'))}",
            f"Sector: {info.get('sector', 'N/A')}",
            f"Industria: {info.get('industry', 'N/A')}",
            f"País: {info.get('country', 'N/A')}",
            f"",
            f"PRECIOS",
            f"Precio actual: ${current_price:.2f}" if isinstance(current_price, (int, float)) else f"Precio actual: {current_price}",
            f"Cambio: ${change:.2f} ({change_pct:.2f}%)" if isinstance(change, (int, float)) else "Cambio: N/A",
            f"Precio anterior: ${prev_close:.2f}" if isinstance(prev_close, (int, float)) else f"Precio anterior: {prev_close}",
            f"Rango 52 semanas: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}",
            f"",
            f"VOLUMEN",
            f"Volumen promedio: {info.get('averageVolume', 'N/A'):,}" if info.get('averageVolume') else "Volumen promedio: N/A",
            f"Volumen actual: {info.get('volume', 'N/A'):,}" if info.get('volume') else "Volumen actual: N/A",
            f"",
            f"CAPITALIZACIÓN",
            f"Market Cap: ${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "Market Cap: N/A",
            f"Shares Outstanding: {info.get('sharesOutstanding', 'N/A'):,}" if info.get('sharesOutstanding') else "Shares Outstanding: N/A",
        ]
        
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "\n".join(lines))
        
        # Gráfico de precios
        self.plot_price_chart(df)
        
    def plot_price_chart(self, df: pd.DataFrame) -> None:
        """Crear gráfico de precios"""
        for widget in self.price_canvas_frame.winfo_children():
            widget.destroy()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Gráfico de precios
        ax1.plot(df.index, df['Close'], label="Precio de cierre", linewidth=2, color='blue')
        ax1.fill_between(df.index, df['Close'], alpha=0.3, color='blue')
        
        # Medias móviles
        if len(df) >= 20:
            ma20 = df['Close'].rolling(window=20).mean()
            ax1.plot(df.index, ma20, label="MA 20", alpha=0.7, color='orange')
            
        if len(df) >= 50:
            ma50 = df['Close'].rolling(window=50).mean()
            ax1.plot(df.index, ma50, label="MA 50", alpha=0.7, color='red')
            
        ax1.set_title(f"{self.current_ticker} - Precio de Cierre", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Precio (USD)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Gráfico de volumen
        ax2.bar(df.index, df['Volume'], alpha=0.7, color='green')
        ax2.set_title("Volumen", fontsize=12)
        ax2.set_xlabel("Fecha", fontsize=12)
        ax2.set_ylabel("Volumen", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.price_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)
        
    def update_tech_tab(self, df: pd.DataFrame) -> None:
        """Actualizar pestaña de análisis técnico"""
        for widget in self.tech_canvas_frame.winfo_children():
            widget.destroy()
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # RSI
        rsi = self.calculate_rsi(df['Close'])
        ax1.plot(df.index, rsi, color='purple', linewidth=2)
        ax1.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Sobrecompra (70)')
        ax1.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Sobreventa (30)')
        ax1.set_title('RSI (14)')
        ax1.set_ylabel('RSI')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 100)
        
        # MACD
        macd, signal, histogram = self.calculate_macd(df['Close'])
        ax2.plot(df.index, macd, label='MACD', color='blue')
        ax2.plot(df.index, signal, label='Signal', color='red')
        ax2.bar(df.index, histogram, label='Histogram', alpha=0.7, color='gray')
        ax2.set_title('MACD')
        ax2.set_ylabel('MACD')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['Close'])
        ax3.plot(df.index, df['Close'], label='Precio', color='blue')
        ax3.plot(df.index, bb_upper, label='BB Superior', color='red', alpha=0.7)
        ax3.plot(df.index, bb_middle, label='BB Medio', color='orange', alpha=0.7)
        ax3.plot(df.index, bb_lower, label='BB Inferior', color='green', alpha=0.7)
        ax3.fill_between(df.index, bb_upper, bb_lower, alpha=0.1, color='gray')
        ax3.set_title('Bollinger Bands')
        ax3.set_ylabel('Precio')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Estocástico
        stoch_k, stoch_d = self.calculate_stochastic(df)
        ax4.plot(df.index, stoch_k, label='%K', color='blue')
        ax4.plot(df.index, stoch_d, label='%D', color='red')
        ax4.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Sobrecompra (80)')
        ax4.axhline(y=20, color='g', linestyle='--', alpha=0.7, label='Sobreventa (20)')
        ax4.set_title('Estocástico')
        ax4.set_ylabel('Estocástico')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.tech_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)
        
    def update_metrics_tab(self, info: dict, df: pd.DataFrame) -> None:
        """Actualizar pestaña de métricas financieras"""
        # Calcular métricas adicionales
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Volatilidad anualizada
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        
        # Análisis de tendencia
        recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        
        lines = [
            f"{'='*60}",
            f"MÉTRICAS FINANCIERAS - {self.current_ticker}",
            f"{'='*60}",
            f"",
            f"VALUACIÓN",
            f"P/E Ratio: {info.get('trailingPE', 'N/A')}",
            f"Forward P/E: {info.get('forwardPE', 'N/A')}",
            f"PEG Ratio: {info.get('pegRatio', 'N/A')}",
            f"Price/Book: {info.get('priceToBook', 'N/A')}",
            f"Price/Sales: {info.get('priceToSalesTrailing12Months', 'N/A')}",
            f"EV/Revenue: {info.get('enterpriseToRevenue', 'N/A')}",
            f"EV/EBITDA: {info.get('enterpriseToEbitda', 'N/A')}",
            f"",
            f"RENTABILIDAD",
            f"ROE: {info.get('returnOnEquity', 'N/A')}",
            f"ROA: {info.get('returnOnAssets', 'N/A')}",
            f"Margen Bruto: {info.get('grossMargins', 'N/A')}",
            f"Margen Operativo: {info.get('operatingMargins', 'N/A')}",
            f"Margen Neto: {info.get('profitMargins', 'N/A')}",
            f"",
            f"CRECIMIENTO",
            f"Crecimiento Ingresos: {info.get('revenueGrowth', 'N/A')}",
            f"Crecimiento Ganancias: {info.get('earningsGrowth', 'N/A')}",
            f"",
            f"DIVIDENDOS",
            f"Dividend Yield: {info.get('dividendYield', 'N/A')}",
            f"Dividend Rate: {info.get('dividendRate', 'N/A')}",
            f"Payout Ratio: {info.get('payoutRatio', 'N/A')}",
            f"",
            f"ANÁLISIS DE RIESGO",
            f"Beta: {info.get('beta', 'N/A')}",
            f"Volatilidad (anualizada): {volatility:.2%}",
            f"Sharpe Ratio: {sharpe_ratio:.2f}",
            f"",
            f"FORTALEZA FINANCIERA",
            f"Debt/Equity: {info.get('debtToEquity', 'N/A')}",
            f"Current Ratio: {info.get('currentRatio', 'N/A')}",
            f"Quick Ratio: {info.get('quickRatio', 'N/A')}",
            f"",
            f"RENDIMIENTO DEL PERÍODO",
            f"Retorno total: {recent_return:.2f}%",
            f"Precio máximo: ${df['High'].max():.2f}",
            f"Precio mínimo: ${df['Low'].min():.2f}",
            f"Volatilidad período: {returns.std():.2%}",
            f"",
            f"RECOMENDACIÓN ANALISTAS",
            f"Recomendación: {info.get('recommendationKey', 'N/A')}",
            f"Precio objetivo: ${info.get('targetMeanPrice', 'N/A')}",
            f"Número de analistas: {info.get('numberOfAnalystOpinions', 'N/A')}",
        ]
        
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, "\n".join(lines))
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calcular MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
        
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calcular Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
        
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        """Calcular Estocástico"""
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
        
    def export_data(self) -> None:
        """Exportar datos a CSV"""
        if self.current_data is None:
            messagebox.showwarning("Aviso", "No hay datos para exportar.")
            return
            
        try:
            filename = f"{self.current_ticker}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.current_data.to_csv(filename)
            messagebox.showinfo("Éxito", f"Datos exportados a {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockAnalyzer(root)
    root.geometry("1200x800")
    root.minsize(1000, 600)
    root.mainloop()