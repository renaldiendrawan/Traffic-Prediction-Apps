# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'traffic_prediction1.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1039, 706)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(820, 390, 161, 51))
        self.pushButton.setStyleSheet("background-color:rgb(255, 0, 0)")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(820, 230, 161, 51))
        self.pushButton_2.setStyleSheet("\n"
"background-color:rgb(0, 255, 0)")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(270, 160, 68, 31))
        self.label.setStyleSheet("color: blue;\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(270, 200, 68, 31))
        self.label_2.setStyleSheet("color: blue;\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(280, 240, 71, 31))
        self.label_3.setStyleSheet("color: blue;\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(200, 280, 131, 31))
        self.label_4.setStyleSheet("color: blue;\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(190, 320, 141, 31))
        self.label_5.setStyleSheet("color: blue;\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(200, 360, 151, 31))
        self.label_6.setStyleSheet("color: blue;\n"
"font: 75 14pt \"MS Shell Dlg 2\";\n"
"")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(170, 400, 151, 31))
        self.label_7.setStyleSheet("color: blue;\n"
"font: 75 14pt \"MS Shell Dlg 2\";\n"
"")
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(260, 450, 68, 31))
        self.label_8.setStyleSheet("color: black;\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.label_8.setObjectName("label_8")
        self.timeEdit = QtWidgets.QTimeEdit(self.centralwidget)
        self.timeEdit.setGeometry(QtCore.QRect(370, 170, 341, 25))
        self.timeEdit.setObjectName("timeEdit")
        self.dateEdit = QtWidgets.QDateEdit(self.centralwidget)
        self.dateEdit.setGeometry(QtCore.QRect(370, 210, 341, 25))
        self.dateEdit.setObjectName("dateEdit")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(370, 250, 341, 25))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(370, 290, 341, 21))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(370, 330, 341, 21))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(370, 370, 341, 21))
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_4 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_4.setGeometry(QtCore.QRect(370, 410, 341, 21))
        self.textEdit_4.setObjectName("textEdit_4")
        self.textEdit_5 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_5.setGeometry(QtCore.QRect(370, 460, 341, 21))
        self.textEdit_5.setObjectName("textEdit_5")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, 0, 1041, 51))
        self.widget.setStyleSheet("background-color:blue;")
        self.widget.setObjectName("widget")
        self.widget_5 = QtWidgets.QWidget(self.widget)
        self.widget_5.setGeometry(QtCore.QRect(40, 10, 41, 31))
        self.widget_5.setObjectName("widget_5")
        self.label_10 = QtWidgets.QLabel(self.widget)
        self.label_10.setGeometry(QtCore.QRect(440, 20, 231, 19))
        self.label_10.setStyleSheet("font: 87 12pt \"Arial Black\"; color : white;")
        self.label_10.setObjectName("label_10")
        self.widget_3 = QtWidgets.QWidget(self.centralwidget)
        self.widget_3.setGeometry(QtCore.QRect(0, 600, 1041, 51))
        self.widget_3.setStyleSheet("background-color:blue;")
        self.widget_3.setObjectName("widget_3")
        self.label_11 = QtWidgets.QLabel(self.widget_3)
        self.label_11.setGeometry(QtCore.QRect(650, 10, 121, 19))
        self.label_11.setStyleSheet("font: 87 7pt \"Arial Black\";")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(440, 70, 191, 71))
        self.label_12.setStyleSheet("font: 75 20pt \"MS Shell Dlg 2\";")
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(60, 210, 53, 16))
        self.label_13.setText("")
        self.label_13.setScaledContents(True)
        self.label_13.setObjectName("label_13")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(820, 310, 161, 51))
        self.pushButton_3.setStyleSheet("background-color:yellow;")
        self.pushButton_3.setObjectName("pushButton_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1039, 31))
        self.menubar.setObjectName("menubar")
        self.menuTraffic_Prediction = QtWidgets.QMenu(self.menubar)
        self.menuTraffic_Prediction.setObjectName("menuTraffic_Prediction")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuTraffic_Prediction.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Clear"))
        self.pushButton_2.setText(_translate("MainWindow", "Process"))
        self.label.setText(_translate("MainWindow", "Time"))
        self.label_2.setText(_translate("MainWindow", "Date"))
        self.label_3.setText(_translate("MainWindow", "Day  "))
        self.label_4.setText(_translate("MainWindow", "Car Count  "))
        self.label_5.setText(_translate("MainWindow", "Bike Count  "))
        self.label_6.setText(_translate("MainWindow", "Bus Count   "))
        self.label_7.setText(_translate("MainWindow", "Truck Count  "))
        self.label_8.setText(_translate("MainWindow", "Total :"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Monday"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Tuesday"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Wednesday"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Thursday"))
        self.comboBox.setItemText(4, _translate("MainWindow", "Friday"))
        self.comboBox.setItemText(5, _translate("MainWindow", "Saturday"))
        self.comboBox.setItemText(6, _translate("MainWindow", "Sunday"))
        self.label_10.setText(_translate("MainWindow", "Traffic Prediction"))
        self.label_12.setText(_translate("MainWindow", "Input Data"))
        self.pushButton_3.setText(_translate("MainWindow", "Model"))
        self.menuTraffic_Prediction.setTitle(_translate("MainWindow", "Traffic Prediction"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
