import sys
import os
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import calendar

# Menentukan base path sesuai lingkungan eksekusi
if getattr(sys, 'frozen', False):
    # Jika dibungkus sebagai .exe oleh PyInstaller
    base_path = sys._MEIPASS
else:
    # Jika dijalankan sebagai script Python
    base_path = os.path.dirname(os.path.abspath(__file__))

# Membaca dataset menggunakan base_path
csv_path = os.path.join(base_path, 'TrafficTwoMonth.csv')
data = pd.read_csv(csv_path, sep=",")

# Mengisi nilai yang hilang pada kolom CarCount dengan rata-rata
data['CarCount'].fillna(data['CarCount'].mean(), inplace=True)

# Mapping kolom 'Traffic Situation' ke nilai numerik
traffic_situation_mapping = {'low': 0, 'normal': 1, 'high': 2, 'heavy': 3}
data['Traffic Situation'] = data['Traffic Situation'].map(traffic_situation_mapping)

# Pisahkan fitur dan target
features = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']
target = 'Traffic Situation'

X = data[features]
y = data[target]

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train Naive Bayes model
nb = GaussianNB()
nb.fit(x_train, y_train)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(x_train, y_train)

# UI setup
from traffic_prediction_ui import Ui_MainWindow  # assuming UI file is named trafficprediction_ui.py

class TrafficPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Set current date and time on startup
        self.set_current_datetime()

        # Connect button clicks to respective functions
        self.ui.pushButton_2.clicked.connect(self.process_prediction)  # Process Button
        self.ui.pushButton.clicked.connect(self.clear_fields)  # Clear Button
        self.ui.pushButton_3.clicked.connect(self.show_model_details)  # Model Button

        # Update day when date is changed
        self.ui.dateEdit.dateChanged.connect(self.update_day_from_date)

    def set_current_datetime(self):
        current_datetime = QtCore.QDateTime.currentDateTime()
        self.ui.timeEdit.setTime(QtCore.QTime.currentTime())
        self.ui.dateEdit.setDate(QtCore.QDate.currentDate())
        self.ui.comboBox.setCurrentText(current_datetime.date().toString("dddd"))  # Set current day

    def process_prediction(self):
        try:
            # Get input from the user interface and validate
            car_count = self.get_int_from_text(self.ui.textEdit.toPlainText(), "Car Count")
            bike_count = self.get_int_from_text(self.ui.textEdit_2.toPlainText(), "Bike Count")
            bus_count = self.get_int_from_text(self.ui.textEdit_3.toPlainText(), "Bus Count")
            truck_count = self.get_int_from_text(self.ui.textEdit_4.toPlainText(), "Truck Count")

            if car_count is None or bike_count is None or bus_count is None or truck_count is None:
                return  # Stop processing if any input is invalid

            # Calculate total count automatically
            total_count = car_count + bike_count + bus_count + truck_count

            # Display the total count in the 'Total' field (textEdit_5)
            self.ui.textEdit_5.setText(str(total_count))

            # Prepare the input data for prediction
            input_data = [[car_count, bike_count, bus_count, truck_count, total_count]]

            # Predict traffic situation using both models
            nb_prediction = nb.predict(input_data)[0]
            rf_prediction = rf.predict(input_data)[0]

            # Calculate accuracy for each model
            nb_accuracy = accuracy_score(y_test, nb.predict(x_test))
            rf_accuracy = accuracy_score(y_test, rf.predict(x_test))

            # Map prediction back to traffic situation
            traffic_situation_reverse_mapping = {0: 'low', 1: 'normal', 2: 'high', 3: 'heavy'}
            predicted_nb_situation = traffic_situation_reverse_mapping[nb_prediction]
            predicted_rf_situation = traffic_situation_reverse_mapping[rf_prediction]

            # Show prediction and accuracy in a message box with total
            msg = QMessageBox()
            msg.setWindowTitle("Traffic Prediction Results")
            msg.setText(f"Total Vehicle Count: {total_count}\n"
                        f"Naive Bayes Prediction: {predicted_nb_situation}\n"
                        f"Naive Bayes Accuracy: {nb_accuracy:.2f}\n"
                        f"Random Forest Prediction: {predicted_rf_situation}\n"
                        f"Random Forest Accuracy: {rf_accuracy:.2f}")
            msg.exec_()

        except Exception as e:
            # If any error occurs, show a message box with the error
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Prediction Error")
            msg.setText(f"An error occurred: {str(e)}")
            msg.exec_()

    def get_int_from_text(self, text, field_name):
        try:
            value = int(text)
            return value
        except ValueError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Input Error")
            msg.setText(f"Please enter a valid integer in the {field_name} field.")
            msg.exec_()
            return None

    def show_model_details(self):
        try:
            car_count = self.get_int_from_text(self.ui.textEdit.toPlainText(), "Car Count")
            bike_count = self.get_int_from_text(self.ui.textEdit_2.toPlainText(), "Bike Count")
            bus_count = self.get_int_from_text(self.ui.textEdit_3.toPlainText(), "Bus Count")
            truck_count = self.get_int_from_text(self.ui.textEdit_4.toPlainText(), "Truck Count")

            if car_count is None or bike_count is None or bus_count is None or truck_count is None:
                return

            total_count = car_count + bike_count + bus_count + truck_count
            self.ui.textEdit_5.setText(str(total_count))

            nb_confusion_matrix = confusion_matrix(y_test, nb.predict(x_test))
            rf_confusion_matrix = confusion_matrix(y_test, rf.predict(x_test))

            nb_accuracy = accuracy_score(y_test, nb.predict(x_test))
            rf_accuracy = accuracy_score(y_test, rf.predict(x_test))

            nb_test_prediction = nb.predict([[car_count, bike_count, bus_count, truck_count, total_count]])[0]
            rf_test_prediction = rf.predict([[car_count, bike_count, bus_count, truck_count, total_count]])[0]

            traffic_situation_reverse_mapping = {0: 'low', 1: 'normal', 2: 'high', 3: 'heavy'}
            nb_prediction_label = traffic_situation_reverse_mapping[nb_test_prediction]
            rf_prediction_label = traffic_situation_reverse_mapping[rf_test_prediction]

            nb_matrix_str = f"Naive Bayes Confusion Matrix:\n{nb_confusion_matrix}"
            rf_matrix_str = f"Random Forest Confusion Matrix:\n{rf_confusion_matrix}"

            msg = QMessageBox()
            msg.setWindowTitle("Model Confusion Matrices, Accuracy, and Prediction")
            msg.setText(f"Total Vehicle Count: {total_count}\n\n"
                        f"Naive Bayes Prediction: {nb_prediction_label}\n"
                        f"Naive Bayes Accuracy: {nb_accuracy:.2f}\n"
                        f"{nb_matrix_str}\n\n"
                        f"Random Forest Prediction: {rf_prediction_label}\n"
                        f"Random Forest Accuracy: {rf_accuracy:.2f}\n"
                        f"{rf_matrix_str}")
            msg.exec_()

        except Exception as e:
            print(f"Error displaying confusion matrix: {e}")

    def clear_fields(self):
        try:
            self.ui.textEdit.clear()
            self.ui.textEdit_2.clear()
            self.ui.textEdit_3.clear()
            self.ui.textEdit_4.clear()
            self.ui.textEdit_5.clear()

            self.ui.timeEdit.setTime(QtCore.QTime.currentTime())
            self.ui.dateEdit.setDate(QtCore.QDate.currentDate())
            self.update_day_from_date()
        except Exception as e:
            print("Error clearing fields:", e)

    def update_day_from_date(self):
        date = self.ui.dateEdit.date().toPyDate()
        day_of_week = calendar.day_name[date.weekday()]
        day_index = self.ui.comboBox.findText(day_of_week)
        if day_index != -1:
            self.ui.comboBox.setCurrentIndex(day_index)

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficPredictionApp()
    window.show()
    sys.exit(app.exec_())
