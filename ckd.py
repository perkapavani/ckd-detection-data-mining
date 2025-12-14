import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tkinter as tk
from tkinter import filedialog, messagebox

class CKDPredictor:
    def __init__(self, root):  # <-- Fixed __init__ typo
        self.root = root
        self.root.title("CKD Risk Predictor")
        self.root.geometry("600x800")
        self.root.configure(bg='#f0f0f0')

        self.data = None
        self.models = {}
        self.scaler = None

        self.create_ui()

    def create_ui(self):
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(
            main_frame,
            text="CKD Risk Predictor",
            font=('Helvetica', 18, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))

        dataset_frame = tk.LabelFrame(main_frame, text="Dataset", font=('Helvetica', 12), bg='#f0f0f0', labelanchor='n')
        dataset_frame.pack(fill=tk.X, pady=10)

        load_btn = tk.Button(dataset_frame, text="Load Dataset", command=self.load_dataset, bg='#3498db', fg='white', font=('Helvetica', 10), relief=tk.FLAT)
        load_btn.pack(pady=10, fill=tk.X, padx=10)

        train_btn = tk.Button(dataset_frame, text="Train Models", command=self.train_models, bg='#2ecc71', fg='white', font=('Helvetica', 10), relief=tk.FLAT)
        train_btn.pack(pady=10, fill=tk.X, padx=10)

        input_frame = tk.LabelFrame(main_frame, text="Patient Details", font=('Helvetica', 12), bg='#f0f0f0', labelanchor='n')
        input_frame.pack(fill=tk.X, pady=10)

        features_left = [
            ('Age', 'age'),
            ('Blood Pressure', 'bp'),
            ('Specific Gravity', 'specific_gravity'),
            ('Albumin', 'albumin')
        ]
        features_right = [
            ('Sugar', 'sugar'),
            ('Blood Urea', 'blood_urea'),
            ('Serum Creatinine', 'serum_creatinine'),
            ('Hemoglobin', 'hemoglobin')
        ]

        columns_frame = tk.Frame(input_frame, bg='#f0f0f0')
        columns_frame.pack(fill=tk.X, padx=10)

        left_column = tk.Frame(columns_frame, bg='#f0f0f0')
        right_column = tk.Frame(columns_frame, bg='#f0f0f0')

        left_column.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        right_column.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5)

        self.feature_entries = {}

        for label, name in features_left:
            row_frame = tk.Frame(left_column, bg='#f0f0f0')
            row_frame.pack(fill=tk.X, pady=5)
            tk.Label(row_frame, text=label, width=25, bg='#f0f0f0', font=('Helvetica', 10)).pack(side=tk.LEFT)
            entry = tk.Entry(row_frame, font=('Helvetica', 10), relief=tk.SOLID, borderwidth=1)
            entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
            self.feature_entries[name] = entry

        for label, name in features_right:
            row_frame = tk.Frame(right_column, bg='#f0f0f0')
            row_frame.pack(fill=tk.X, pady=5)
            tk.Label(row_frame, text=label, width=25, bg='#f0f0f0', font=('Helvetica', 10)).pack(side=tk.LEFT)
            entry = tk.Entry(row_frame, font=('Helvetica', 10), relief=tk.SOLID, borderwidth=1)
            entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
            self.feature_entries[name] = entry

        algo_frame = tk.LabelFrame(main_frame, text="Select Algorithm", font=('Helvetica', 12), bg='#f0f0f0', labelanchor='n')
        algo_frame.pack(fill=tk.X, pady=10)

        self.algorithm_var = tk.StringVar(value="Random Forest")
        algorithms = ["Random Forest", "SVM", "KNN"]

        for algo in algorithms:
            radio = tk.Radiobutton(algo_frame, text=algo, variable=self.algorithm_var, value=algo, bg='#f0f0f0', font=('Helvetica', 10))
            radio.pack(anchor='w', padx=10, pady=5)

        predict_btn = tk.Button(main_frame, text="Predict CKD Risk", command=self.predict_risk, bg='#e74c3c', fg='white', font=('Helvetica', 12, 'bold'), relief=tk.FLAT)
        predict_btn.pack(pady=20, fill=tk.X, padx=10)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(title="Select Dataset", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)

            # Fill missing numeric values
            numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
            self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].mean())

            # Fill missing categorical values
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
                self.data[column] = pd.to_numeric(self.data[column], errors='coerce')

            messagebox.showinfo("Dataset Loaded", f"Dataset loaded successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
        else:
            messagebox.showwarning("No File Selected", "Please select a valid dataset file.")

    def validate_inputs(self):
        try:
            features = [
                float(self.feature_entries[name].get())
                for name in ['age', 'bp', 'specific_gravity', 'albumin', 'sugar', 'blood_urea', 'serum_creatinine', 'hemoglobin']
            ]
            return features
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
            return None

    def train_models(self):
        if self.data is None:
            messagebox.showwarning("No Dataset", "Please load a dataset first!")
            return

        try:
            if 'CKD' not in self.data.columns:
                raise ValueError("Dataset must have a 'CKD' column as target.")

            X = self.data.drop(columns=['CKD'])
            y = self.data['CKD']

            # Encode target if necessary
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            self.models = {
                'Random Forest': RandomForestClassifier().fit(X_train, y_train),
                'SVM': SVC(probability=True).fit(X_train, y_train),
                'KNN': KNeighborsClassifier().fit(X_train, y_train)
            }

            metrics = []
            for name, model in self.models.items():
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                metrics.append(f"{name}: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}")

            messagebox.showinfo("Training Complete", "Models trained successfully!\n\n" + "\n".join(metrics))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while training: {e}")

    def predict_risk(self):
        if not self.models or self.scaler is None:
            messagebox.showwarning("Not Ready", "Please load dataset and train models first!")
            return

        features = self.validate_inputs()
        if features is None:
            return

        try:
            scaled_features = self.scaler.transform([features])
            selected_model = self.algorithm_var.get()
            model = self.models[selected_model]
            prediction = model.predict(scaled_features)[0]

            result = "High Risk of CKD" if prediction == 1 else "Low Risk of CKD"
            messagebox.showinfo("Prediction Result", f"The person is at: {result}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")

def main():
    root = tk.Tk()
    app = CKDPredictor(root)
    root.mainloop()

# Fixed main function check
if __name__ == "__main__":
    main()
