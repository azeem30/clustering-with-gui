import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class KMeansClusteringApp:
    def __init__(self, master):
        self.master = master
        self.master.title("K-Means Clustering App")

        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=10, pady=10)

        self.label_text = tk.StringVar(value="Select a .csv file")
        self.label = tk.Label(self.frame, textvariable=self.label_text, font=("Arial", 12))
        self.label.pack()

        self.browse_button = tk.Button(self.frame, text="Browse", command=self.browse_file, font=("Arial", 10), borderwidth=3, relief=tk.RAISED)
        self.browse_button.pack(pady=5)

        self.cluster_button = tk.Button(self.frame, text="Generate Clusters", command=self.perform_clustering, font=("Arial", 10), borderwidth=3, relief=tk.RAISED)
        self.cluster_button.pack(pady=5)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(padx=10, pady=10)

    def browse_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filename:
            self.label_text.set(f"Selected file: {self.filename}")

            self.data = pd.read_csv(self.filename)
            self.data.dropna(inplace=True)
            self.features = list(self.data.columns)
            self.features_var = tk.StringVar(value=self.features)

            self.features_label = tk.Label(self.frame, text="Select features for clustering:", font=("Arial", 10))
            self.features_label.pack()

            self.features_listbox = tk.Listbox(self.frame, listvariable=self.features_var, selectmode=tk.MULTIPLE, height=5, font=("Arial", 10))
            self.features_listbox.pack()

    def get_num_clusters(self):
        return simpledialog.askinteger("Input", "Enter the number of clusters:", parent=self.master, minvalue=1)

    def perform_clustering(self):
        try:
            if not hasattr(self, 'filename'):
                messagebox.showwarning("Warning", "Please select a .csv file first.")
                return

            selected_indices = self.features_listbox.curselection()
            if len(selected_indices) < 2:
                messagebox.showwarning("Warning", "Please select at least two features.")
                return

            selected_features = [self.features[i] for i in selected_indices]
            data = self.data[selected_features]

            num_clusters = self.get_num_clusters()
            if num_clusters is None:
                return

            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(data)
            clusters = kmeans.predict(data)

            self.ax.clear()
            self.ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters, cmap='viridis')
            self.ax.set_title('K-Means Clustering', fontsize=14)
            self.ax.set_xlabel(selected_features[0], fontsize=12)
            self.ax.set_ylabel(selected_features[1], fontsize=12)
            self.ax.tick_params(axis='both', which='major', labelsize=10)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    root.geometry("600x680")
    root.resizable(False, False)
    KMeansClusteringApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
