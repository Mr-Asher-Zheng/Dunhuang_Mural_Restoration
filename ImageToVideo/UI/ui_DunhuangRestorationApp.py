# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'DunhuangRestorationApp.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGraphicsView,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMenu, QMenuBar,
    QPushButton, QScrollArea, QSizePolicy, QSpacerItem,
    QStatusBar, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1126, 635)
        self.actionby_Asher = QAction(MainWindow)
        self.actionby_Asher.setObjectName(u"actionby_Asher")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setMinimumSize(QSize(0, 0))
        self.verticalLayout_8 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        font = QFont()
        font.setPointSize(11)
        self.tabWidget.setFont(font)
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.verticalLayout_22 = QVBoxLayout(self.tab_3)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.label_6 = QLabel(self.tab_3)
        self.label_6.setObjectName(u"label_6")
        font1 = QFont()
        font1.setFamilies([u"\u5b8b\u4f53"])
        font1.setPointSize(32)
        font1.setBold(True)
        self.label_6.setFont(font1)
        self.label_6.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_22.addWidget(self.label_6)

        self.tabWidget.addTab(self.tab_3, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.verticalLayout_7 = QVBoxLayout(self.tab)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.scrollArea = QScrollArea(self.tab)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 1070, 902))
        self.gridLayout_2 = QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.groupBox_Seg = QGroupBox(self.scrollAreaWidgetContents_2)
        self.groupBox_Seg.setObjectName(u"groupBox_Seg")
        self.groupBox_Seg.setMinimumSize(QSize(0, 200))
        font2 = QFont()
        font2.setPointSize(16)
        font2.setBold(True)
        self.groupBox_Seg.setFont(font2)
        self.verticalLayout = QVBoxLayout(self.groupBox_Seg)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_2, 0, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer, 0, 2, 1, 1)

        self.pushButton_SAM_base_model_chose = QPushButton(self.groupBox_Seg)
        self.pushButton_SAM_base_model_chose.setObjectName(u"pushButton_SAM_base_model_chose")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_SAM_base_model_chose.sizePolicy().hasHeightForWidth())
        self.pushButton_SAM_base_model_chose.setSizePolicy(sizePolicy)
        self.pushButton_SAM_base_model_chose.setMinimumSize(QSize(60, 40))
        font3 = QFont()
        font3.setPointSize(14)
        font3.setBold(False)
        self.pushButton_SAM_base_model_chose.setFont(font3)

        self.gridLayout_4.addWidget(self.pushButton_SAM_base_model_chose, 0, 3, 1, 1)

        self.lineEdit_SAM_finetune_model_path = QLineEdit(self.groupBox_Seg)
        self.lineEdit_SAM_finetune_model_path.setObjectName(u"lineEdit_SAM_finetune_model_path")
        sizePolicy.setHeightForWidth(self.lineEdit_SAM_finetune_model_path.sizePolicy().hasHeightForWidth())
        self.lineEdit_SAM_finetune_model_path.setSizePolicy(sizePolicy)
        self.lineEdit_SAM_finetune_model_path.setMinimumSize(QSize(0, 30))
        self.lineEdit_SAM_finetune_model_path.setFont(font3)
        self.lineEdit_SAM_finetune_model_path.setReadOnly(True)

        self.gridLayout_4.addWidget(self.lineEdit_SAM_finetune_model_path, 1, 1, 1, 1)

        self.pushButton_SAM_finetune_model_chose = QPushButton(self.groupBox_Seg)
        self.pushButton_SAM_finetune_model_chose.setObjectName(u"pushButton_SAM_finetune_model_chose")
        sizePolicy.setHeightForWidth(self.pushButton_SAM_finetune_model_chose.sizePolicy().hasHeightForWidth())
        self.pushButton_SAM_finetune_model_chose.setSizePolicy(sizePolicy)
        self.pushButton_SAM_finetune_model_chose.setMinimumSize(QSize(60, 40))
        self.pushButton_SAM_finetune_model_chose.setFont(font3)

        self.gridLayout_4.addWidget(self.pushButton_SAM_finetune_model_chose, 1, 3, 1, 1)

        self.lineEdit_SAM_base_model_path = QLineEdit(self.groupBox_Seg)
        self.lineEdit_SAM_base_model_path.setObjectName(u"lineEdit_SAM_base_model_path")
        sizePolicy.setHeightForWidth(self.lineEdit_SAM_base_model_path.sizePolicy().hasHeightForWidth())
        self.lineEdit_SAM_base_model_path.setSizePolicy(sizePolicy)
        self.lineEdit_SAM_base_model_path.setMinimumSize(QSize(0, 30))
        self.lineEdit_SAM_base_model_path.setFont(font3)
        self.lineEdit_SAM_base_model_path.setReadOnly(True)

        self.gridLayout_4.addWidget(self.lineEdit_SAM_base_model_path, 0, 1, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_3, 0, 4, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout_4)


        self.verticalLayout_2.addWidget(self.groupBox_Seg, 0, Qt.AlignmentFlag.AlignVCenter)

        self.groupBox_2 = QGroupBox(self.scrollAreaWidgetContents_2)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setMinimumSize(QSize(0, 200))
        self.groupBox_2.setFont(font2)
        self.verticalLayout_17 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.gridLayout_6 = QGridLayout()
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.pushButton_sd_model_chose = QPushButton(self.groupBox_2)
        self.pushButton_sd_model_chose.setObjectName(u"pushButton_sd_model_chose")
        sizePolicy.setHeightForWidth(self.pushButton_sd_model_chose.sizePolicy().hasHeightForWidth())
        self.pushButton_sd_model_chose.setSizePolicy(sizePolicy)
        self.pushButton_sd_model_chose.setMinimumSize(QSize(60, 40))
        self.pushButton_sd_model_chose.setFont(font3)

        self.gridLayout_6.addWidget(self.pushButton_sd_model_chose, 0, 3, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer_4, 0, 0, 1, 1)

        self.lineEdit_controlnet_model_path = QLineEdit(self.groupBox_2)
        self.lineEdit_controlnet_model_path.setObjectName(u"lineEdit_controlnet_model_path")
        sizePolicy.setHeightForWidth(self.lineEdit_controlnet_model_path.sizePolicy().hasHeightForWidth())
        self.lineEdit_controlnet_model_path.setSizePolicy(sizePolicy)
        self.lineEdit_controlnet_model_path.setMinimumSize(QSize(0, 30))
        self.lineEdit_controlnet_model_path.setFont(font3)
        self.lineEdit_controlnet_model_path.setReadOnly(True)

        self.gridLayout_6.addWidget(self.lineEdit_controlnet_model_path, 1, 1, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer_5, 0, 2, 1, 1)

        self.pushButton_controlnet_model_chose = QPushButton(self.groupBox_2)
        self.pushButton_controlnet_model_chose.setObjectName(u"pushButton_controlnet_model_chose")
        sizePolicy.setHeightForWidth(self.pushButton_controlnet_model_chose.sizePolicy().hasHeightForWidth())
        self.pushButton_controlnet_model_chose.setSizePolicy(sizePolicy)
        self.pushButton_controlnet_model_chose.setMinimumSize(QSize(60, 40))
        self.pushButton_controlnet_model_chose.setFont(font3)

        self.gridLayout_6.addWidget(self.pushButton_controlnet_model_chose, 1, 3, 1, 1)

        self.lineEdit_sd_model_path = QLineEdit(self.groupBox_2)
        self.lineEdit_sd_model_path.setObjectName(u"lineEdit_sd_model_path")
        sizePolicy.setHeightForWidth(self.lineEdit_sd_model_path.sizePolicy().hasHeightForWidth())
        self.lineEdit_sd_model_path.setSizePolicy(sizePolicy)
        self.lineEdit_sd_model_path.setMinimumSize(QSize(0, 30))
        self.lineEdit_sd_model_path.setFont(font3)
        self.lineEdit_sd_model_path.setReadOnly(True)

        self.gridLayout_6.addWidget(self.lineEdit_sd_model_path, 0, 1, 1, 1)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer_6, 0, 4, 1, 1)


        self.verticalLayout_17.addLayout(self.gridLayout_6)


        self.verticalLayout_2.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.scrollAreaWidgetContents_2)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setMinimumSize(QSize(0, 200))
        self.groupBox_3.setFont(font2)
        self.verticalLayout_21 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.gridLayout_12 = QGridLayout()
        self.gridLayout_12.setObjectName(u"gridLayout_12")
        self.horizontalSpacer_16 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_12.addItem(self.horizontalSpacer_16, 0, 0, 1, 1)

        self.horizontalSpacer_15 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_12.addItem(self.horizontalSpacer_15, 0, 2, 1, 1)

        self.lineEdit_sr_model_path = QLineEdit(self.groupBox_3)
        self.lineEdit_sr_model_path.setObjectName(u"lineEdit_sr_model_path")
        sizePolicy.setHeightForWidth(self.lineEdit_sr_model_path.sizePolicy().hasHeightForWidth())
        self.lineEdit_sr_model_path.setSizePolicy(sizePolicy)
        self.lineEdit_sr_model_path.setMinimumSize(QSize(60, 30))
        self.lineEdit_sr_model_path.setFont(font3)
        self.lineEdit_sr_model_path.setReadOnly(True)

        self.gridLayout_12.addWidget(self.lineEdit_sr_model_path, 0, 1, 1, 1)

        self.pushButton_sr_model_chose = QPushButton(self.groupBox_3)
        self.pushButton_sr_model_chose.setObjectName(u"pushButton_sr_model_chose")
        sizePolicy.setHeightForWidth(self.pushButton_sr_model_chose.sizePolicy().hasHeightForWidth())
        self.pushButton_sr_model_chose.setSizePolicy(sizePolicy)
        self.pushButton_sr_model_chose.setMinimumSize(QSize(60, 40))
        self.pushButton_sr_model_chose.setFont(font3)

        self.gridLayout_12.addWidget(self.pushButton_sr_model_chose, 0, 3, 1, 1)

        self.horizontalSpacer_17 = QSpacerItem(40, 20, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_12.addItem(self.horizontalSpacer_17, 0, 4, 1, 1)


        self.verticalLayout_21.addLayout(self.gridLayout_12)


        self.verticalLayout_2.addWidget(self.groupBox_3)

        self.groupBox_4 = QGroupBox(self.scrollAreaWidgetContents_2)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setMinimumSize(QSize(0, 100))
        self.groupBox_4.setFont(font2)
        self.verticalLayout_9 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.textEdit_api_key = QTextEdit(self.groupBox_4)
        self.textEdit_api_key.setObjectName(u"textEdit_api_key")
        self.textEdit_api_key.setFont(font3)

        self.verticalLayout_9.addWidget(self.textEdit_api_key)


        self.verticalLayout_2.addWidget(self.groupBox_4)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.pushButton_application = QPushButton(self.scrollAreaWidgetContents_2)
        self.pushButton_application.setObjectName(u"pushButton_application")
        self.pushButton_application.setMinimumSize(QSize(0, 50))
        self.pushButton_application.setFont(font3)

        self.horizontalLayout.addWidget(self.pushButton_application)

        self.pushButton_clean = QPushButton(self.scrollAreaWidgetContents_2)
        self.pushButton_clean.setObjectName(u"pushButton_clean")
        self.pushButton_clean.setMinimumSize(QSize(0, 50))
        self.pushButton_clean.setFont(font3)

        self.horizontalLayout.addWidget(self.pushButton_clean)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.groupBox_5 = QGroupBox(self.scrollAreaWidgetContents_2)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setMinimumSize(QSize(0, 100))
        self.groupBox_5.setFont(font2)
        self.verticalLayout_6 = QVBoxLayout(self.groupBox_5)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.scrollArea_3 = QScrollArea(self.groupBox_5)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_4 = QWidget()
        self.scrollAreaWidgetContents_4.setObjectName(u"scrollAreaWidgetContents_4")
        self.scrollAreaWidgetContents_4.setGeometry(QRect(0, 0, 1016, 74))
        self.verticalLayout_5 = QVBoxLayout(self.scrollAreaWidgetContents_4)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.textEdit_status = QTextEdit(self.scrollAreaWidgetContents_4)
        self.textEdit_status.setObjectName(u"textEdit_status")
        self.textEdit_status.setMinimumSize(QSize(0, 0))
        self.textEdit_status.setMaximumSize(QSize(16777215, 75))
        self.textEdit_status.setFont(font3)
        self.textEdit_status.setReadOnly(True)

        self.verticalLayout_5.addWidget(self.textEdit_status)

        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_4)

        self.verticalLayout_6.addWidget(self.scrollArea_3)


        self.verticalLayout_2.addWidget(self.groupBox_5)


        self.gridLayout_2.addLayout(self.verticalLayout_2, 0, 0, 1, 1)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_7.addWidget(self.scrollArea)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_10 = QVBoxLayout(self.tab_2)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.scrollArea_2 = QScrollArea(self.tab_2)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setMaximumSize(QSize(1600, 800))
        self.scrollArea_2.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scrollArea_2.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, -3530, 1070, 4042))
        self.gridLayout_3 = QGridLayout(self.scrollAreaWidgetContents_3)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.groupBox = QGroupBox(self.scrollAreaWidgetContents_3)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setMinimumSize(QSize(500, 800))
        self.groupBox.setFont(font2)
        self.verticalLayout_4 = QVBoxLayout(self.groupBox)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(0, 100))
        self.label_2.setFont(font3)

        self.gridLayout_5.addWidget(self.label_2, 0, 0, 1, 10)

        self.frame_2 = QFrame(self.groupBox)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy1)
        self.frame_2.setMinimumSize(QSize(512, 512))
        self.frame_2.setStyleSheet(u"    border: 2px solid #555;       /* \u8fb9\u6846\u7c97\u7ec6 + \u989c\u8272 */\n"
"    border-radius: 5px;           /* \u5706\u89d2 */\n"
"    background-color: #1e1e1e;    /* \u80cc\u666f\u8272 */")
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_7 = QGridLayout(self.frame_2)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.pushButton_pointer = QPushButton(self.frame_2)
        self.pushButton_pointer.setObjectName(u"pushButton_pointer")
        self.pushButton_pointer.setFont(font3)
        self.pushButton_pointer.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.gridLayout_7.addWidget(self.pushButton_pointer, 1, 0, 1, 1)

        self.pushButton_line_art_extraction = QPushButton(self.frame_2)
        self.pushButton_line_art_extraction.setObjectName(u"pushButton_line_art_extraction")
        self.pushButton_line_art_extraction.setFont(font3)
        self.pushButton_line_art_extraction.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.gridLayout_7.addWidget(self.pushButton_line_art_extraction, 0, 0, 1, 1)

        self.pushButton_paintbrush = QPushButton(self.frame_2)
        self.pushButton_paintbrush.setObjectName(u"pushButton_paintbrush")
        self.pushButton_paintbrush.setFont(font3)
        self.pushButton_paintbrush.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.gridLayout_7.addWidget(self.pushButton_paintbrush, 1, 1, 1, 1)

        self.pushButton_eraser = QPushButton(self.frame_2)
        self.pushButton_eraser.setObjectName(u"pushButton_eraser")
        self.pushButton_eraser.setFont(font3)
        self.pushButton_eraser.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.gridLayout_7.addWidget(self.pushButton_eraser, 1, 2, 1, 1)

        self.pushButton_upload_line = QPushButton(self.frame_2)
        self.pushButton_upload_line.setObjectName(u"pushButton_upload_line")
        self.pushButton_upload_line.setFont(font3)
        self.pushButton_upload_line.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.gridLayout_7.addWidget(self.pushButton_upload_line, 0, 2, 1, 1)

        self.graphicsView_line = QGraphicsView(self.frame_2)
        self.graphicsView_line.setObjectName(u"graphicsView_line")

        self.gridLayout_7.addWidget(self.graphicsView_line, 2, 0, 1, 3)


        self.gridLayout_5.addWidget(self.frame_2, 3, 0, 1, 5)

        self.frame_4 = QFrame(self.groupBox)
        self.frame_4.setObjectName(u"frame_4")
        sizePolicy1.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy1)
        self.frame_4.setMinimumSize(QSize(512, 512))
        self.frame_4.setStyleSheet(u"    border: 2px solid #555;       /* \u8fb9\u6846\u7c97\u7ec6 + \u989c\u8272 */\n"
"    border-radius: 5px;           /* \u5706\u89d2 */\n"
"    background-color: #1e1e1e;    /* \u80cc\u666f\u8272 */")
        self.frame_4.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_8 = QGridLayout(self.frame_4)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.pushButton_auto_mask = QPushButton(self.frame_4)
        self.pushButton_auto_mask.setObjectName(u"pushButton_auto_mask")
        self.pushButton_auto_mask.setFont(font3)
        self.pushButton_auto_mask.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.gridLayout_8.addWidget(self.pushButton_auto_mask, 0, 1, 1, 1)

        self.pushButton_mask_paintbrush = QPushButton(self.frame_4)
        self.pushButton_mask_paintbrush.setObjectName(u"pushButton_mask_paintbrush")
        self.pushButton_mask_paintbrush.setFont(font3)
        self.pushButton_mask_paintbrush.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.gridLayout_8.addWidget(self.pushButton_mask_paintbrush, 1, 2, 1, 1)

        self.pushButton_mask_eraser = QPushButton(self.frame_4)
        self.pushButton_mask_eraser.setObjectName(u"pushButton_mask_eraser")
        self.pushButton_mask_eraser.setFont(font3)
        self.pushButton_mask_eraser.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.gridLayout_8.addWidget(self.pushButton_mask_eraser, 1, 3, 1, 1)

        self.pushButton_pointer_2 = QPushButton(self.frame_4)
        self.pushButton_pointer_2.setObjectName(u"pushButton_pointer_2")
        self.pushButton_pointer_2.setFont(font3)
        self.pushButton_pointer_2.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.gridLayout_8.addWidget(self.pushButton_pointer_2, 1, 1, 1, 1)

        self.pushButton_all_mask = QPushButton(self.frame_4)
        self.pushButton_all_mask.setObjectName(u"pushButton_all_mask")
        self.pushButton_all_mask.setFont(font3)
        self.pushButton_all_mask.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.gridLayout_8.addWidget(self.pushButton_all_mask, 0, 2, 1, 1)

        self.pushButton_clean_all_mask = QPushButton(self.frame_4)
        self.pushButton_clean_all_mask.setObjectName(u"pushButton_clean_all_mask")
        self.pushButton_clean_all_mask.setFont(font3)
        self.pushButton_clean_all_mask.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.gridLayout_8.addWidget(self.pushButton_clean_all_mask, 0, 3, 1, 1)

        self.graphicsView_mask_manual = QGraphicsView(self.frame_4)
        self.graphicsView_mask_manual.setObjectName(u"graphicsView_mask_manual")

        self.gridLayout_8.addWidget(self.graphicsView_mask_manual, 2, 1, 1, 3)


        self.gridLayout_5.addWidget(self.frame_4, 3, 5, 1, 5)

        self.pushButton_export_line = QPushButton(self.groupBox)
        self.pushButton_export_line.setObjectName(u"pushButton_export_line")
        self.pushButton_export_line.setMinimumSize(QSize(0, 50))
        self.pushButton_export_line.setFont(font3)

        self.gridLayout_5.addWidget(self.pushButton_export_line, 2, 0, 1, 5)

        self.pushButton_export_mask = QPushButton(self.groupBox)
        self.pushButton_export_mask.setObjectName(u"pushButton_export_mask")
        self.pushButton_export_mask.setMinimumSize(QSize(0, 50))
        self.pushButton_export_mask.setFont(font3)

        self.gridLayout_5.addWidget(self.pushButton_export_mask, 2, 5, 1, 5)


        self.verticalLayout_4.addLayout(self.gridLayout_5)


        self.gridLayout_3.addWidget(self.groupBox, 1, 0, 1, 1)

        self.groupBox_7 = QGroupBox(self.scrollAreaWidgetContents_3)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.groupBox_7.setMinimumSize(QSize(500, 800))
        self.groupBox_7.setFont(font2)
        self.verticalLayout_13 = QVBoxLayout(self.groupBox_7)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.label_3 = QLabel(self.groupBox_7)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(0, 100))
        self.label_3.setFont(font3)

        self.verticalLayout_13.addWidget(self.label_3)

        self.gridLayout_9 = QGridLayout()
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.frame_5 = QFrame(self.groupBox_7)
        self.frame_5.setObjectName(u"frame_5")
        sizePolicy1.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy1)
        self.frame_5.setMinimumSize(QSize(512, 512))
        self.frame_5.setStyleSheet(u"    border: 2px solid #555;       /* \u8fb9\u6846\u7c97\u7ec6 + \u989c\u8272 */\n"
"    border-radius: 5px;           /* \u5706\u89d2 */\n"
"    background-color: #1e1e1e;    /* \u80cc\u666f\u8272 */")
        self.frame_5.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_15 = QVBoxLayout(self.frame_5)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.graphicsView_image_restoration = QGraphicsView(self.frame_5)
        self.graphicsView_image_restoration.setObjectName(u"graphicsView_image_restoration")

        self.verticalLayout_15.addWidget(self.graphicsView_image_restoration)


        self.gridLayout_9.addWidget(self.frame_5, 0, 1, 3, 1)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_9.addItem(self.horizontalSpacer_10, 0, 2, 1, 1)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_9.addItem(self.horizontalSpacer_11, 0, 4, 1, 1)

        self.pushButton_export_restore_image = QPushButton(self.groupBox_7)
        self.pushButton_export_restore_image.setObjectName(u"pushButton_export_restore_image")
        self.pushButton_export_restore_image.setMinimumSize(QSize(0, 50))
        self.pushButton_export_restore_image.setFont(font3)

        self.gridLayout_9.addWidget(self.pushButton_export_restore_image, 2, 3, 1, 1)

        self.lineEdit_step = QLineEdit(self.groupBox_7)
        self.lineEdit_step.setObjectName(u"lineEdit_step")
        self.lineEdit_step.setMinimumSize(QSize(0, 50))
        self.lineEdit_step.setFont(font3)

        self.gridLayout_9.addWidget(self.lineEdit_step, 0, 3, 1, 1)

        self.pushButton_image_restoration = QPushButton(self.groupBox_7)
        self.pushButton_image_restoration.setObjectName(u"pushButton_image_restoration")
        self.pushButton_image_restoration.setMinimumSize(QSize(0, 50))
        self.pushButton_image_restoration.setFont(font3)

        self.gridLayout_9.addWidget(self.pushButton_image_restoration, 1, 3, 1, 1)

        self.horizontalSpacer_14 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_9.addItem(self.horizontalSpacer_14, 0, 0, 1, 1)


        self.verticalLayout_13.addLayout(self.gridLayout_9)


        self.gridLayout_3.addWidget(self.groupBox_7, 2, 0, 1, 1)

        self.groupBox_6 = QGroupBox(self.scrollAreaWidgetContents_3)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setMinimumSize(QSize(500, 800))
        self.groupBox_6.setFont(font2)
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_6)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_5, 4, 1, 1, 1)

        self.frame = QFrame(self.groupBox_6)
        self.frame.setObjectName(u"frame")
        sizePolicy1.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy1)
        self.frame.setMinimumSize(QSize(512, 512))
        self.frame.setStyleSheet(u"    border: 2px solid #555;       /* \u8fb9\u6846\u7c97\u7ec6 + \u989c\u8272 */\n"
"    border-radius: 5px;           /* \u5706\u89d2 */\n"
"    background-color: #1e1e1e;    /* \u80cc\u666f\u8272 */")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_11 = QVBoxLayout(self.frame)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.graphicsView_init_image = QGraphicsView(self.frame)
        self.graphicsView_init_image.setObjectName(u"graphicsView_init_image")

        self.verticalLayout_11.addWidget(self.graphicsView_init_image)


        self.gridLayout.addWidget(self.frame, 3, 1, 1, 1)

        self.pushButton_upload_image = QPushButton(self.groupBox_6)
        self.pushButton_upload_image.setObjectName(u"pushButton_upload_image")
        self.pushButton_upload_image.setMinimumSize(QSize(200, 50))
        self.pushButton_upload_image.setFont(font3)

        self.gridLayout.addWidget(self.pushButton_upload_image, 7, 1, 1, 1)

        self.verticalSpacer_7 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_7, 8, 1, 1, 1)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_7, 3, 2, 1, 1)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_4, 2, 1, 1, 1)

        self.pushButton_seg_detect = QPushButton(self.groupBox_6)
        self.pushButton_seg_detect.setObjectName(u"pushButton_seg_detect")
        self.pushButton_seg_detect.setMinimumSize(QSize(200, 50))
        self.pushButton_seg_detect.setFont(font3)

        self.gridLayout.addWidget(self.pushButton_seg_detect, 7, 3, 1, 1)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_8, 3, 0, 1, 1)

        self.lineEdit_device = QLineEdit(self.groupBox_6)
        self.lineEdit_device.setObjectName(u"lineEdit_device")
        self.lineEdit_device.setMinimumSize(QSize(200, 50))
        self.lineEdit_device.setFont(font3)

        self.gridLayout.addWidget(self.lineEdit_device, 5, 3, 1, 1)

        self.lineEdit_size = QLineEdit(self.groupBox_6)
        self.lineEdit_size.setObjectName(u"lineEdit_size")
        self.lineEdit_size.setMinimumSize(QSize(200, 50))
        self.lineEdit_size.setFont(font3)

        self.gridLayout.addWidget(self.lineEdit_size, 5, 1, 1, 1)

        self.frame_3 = QFrame(self.groupBox_6)
        self.frame_3.setObjectName(u"frame_3")
        sizePolicy1.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy1)
        self.frame_3.setMinimumSize(QSize(512, 512))
        self.frame_3.setStyleSheet(u"    border: 2px solid #555;       /* \u8fb9\u6846\u7c97\u7ec6 + \u989c\u8272 */\n"
"    border-radius: 5px;           /* \u5706\u89d2 */\n"
"    background-color: #1e1e1e;    /* \u80cc\u666f\u8272 */")
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_12 = QVBoxLayout(self.frame_3)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.graphicsView_mask = QGraphicsView(self.frame_3)
        self.graphicsView_mask.setObjectName(u"graphicsView_mask")

        self.verticalLayout_12.addWidget(self.graphicsView_mask)


        self.gridLayout.addWidget(self.frame_3, 3, 3, 1, 1)

        self.verticalSpacer_6 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_6, 0, 1, 1, 1)

        self.label = QLabel(self.groupBox_6)
        self.label.setObjectName(u"label")
        self.label.setMinimumSize(QSize(0, 100))
        self.label.setFont(font3)

        self.gridLayout.addWidget(self.label, 1, 1, 1, 3)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_9, 3, 4, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 6, 1, 1, 1)


        self.verticalLayout_3.addLayout(self.gridLayout)


        self.gridLayout_3.addWidget(self.groupBox_6, 0, 0, 1, 1)

        self.groupBox_9 = QGroupBox(self.scrollAreaWidgetContents_3)
        self.groupBox_9.setObjectName(u"groupBox_9")
        self.groupBox_9.setMinimumSize(QSize(500, 800))
        self.groupBox_9.setFont(font2)
        self.verticalLayout_14 = QVBoxLayout(self.groupBox_9)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.gridLayout_10 = QGridLayout()
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.comboBox_model = QComboBox(self.groupBox_9)
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_model.setObjectName(u"comboBox_model")
        self.comboBox_model.setMinimumSize(QSize(0, 50))
        self.comboBox_model.setFont(font3)

        self.gridLayout_10.addWidget(self.comboBox_model, 2, 1, 1, 2)

        self.label_4 = QLabel(self.groupBox_9)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMinimumSize(QSize(0, 100))
        self.label_4.setFont(font3)

        self.gridLayout_10.addWidget(self.label_4, 0, 0, 1, 3)

        self.pushButton_image_to_video = QPushButton(self.groupBox_9)
        self.pushButton_image_to_video.setObjectName(u"pushButton_image_to_video")
        self.pushButton_image_to_video.setMinimumSize(QSize(0, 50))
        self.pushButton_image_to_video.setFont(font3)

        self.gridLayout_10.addWidget(self.pushButton_image_to_video, 5, 1, 1, 2)

        self.frame_6 = QFrame(self.groupBox_9)
        self.frame_6.setObjectName(u"frame_6")
        sizePolicy1.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy1)
        self.frame_6.setMinimumSize(QSize(512, 512))
        self.frame_6.setStyleSheet(u"    border: 2px solid #555;       /* \u8fb9\u6846\u7c97\u7ec6 + \u989c\u8272 */\n"
"    border-radius: 5px;           /* \u5706\u89d2 */\n"
"    background-color: #1e1e1e;    /* \u80cc\u666f\u8272 */")
        self.frame_6.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_16 = QVBoxLayout(self.frame_6)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSpacer_13 = QSpacerItem(40, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_13)

        self.pushButton_play_pause = QPushButton(self.frame_6)
        self.pushButton_play_pause.setObjectName(u"pushButton_play_pause")
        sizePolicy.setHeightForWidth(self.pushButton_play_pause.sizePolicy().hasHeightForWidth())
        self.pushButton_play_pause.setSizePolicy(sizePolicy)
        self.pushButton_play_pause.setMinimumSize(QSize(0, 40))
        self.pushButton_play_pause.setMaximumSize(QSize(500, 16777215))
        self.pushButton_play_pause.setFont(font)
        self.pushButton_play_pause.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.pushButton_play_pause.setStyleSheet(u"    background-color: rgb(255, 255, 255);    /* \u80cc\u666f\u8272 */")

        self.horizontalLayout_3.addWidget(self.pushButton_play_pause)

        self.horizontalSpacer_12 = QSpacerItem(40, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_12)


        self.verticalLayout_16.addLayout(self.horizontalLayout_3)

        self.graphicsView_video = QGraphicsView(self.frame_6)
        self.graphicsView_video.setObjectName(u"graphicsView_video")
        self.graphicsView_video.setMinimumSize(QSize(0, 0))

        self.verticalLayout_16.addWidget(self.graphicsView_video)


        self.gridLayout_10.addWidget(self.frame_6, 1, 0, 5, 1)

        self.comboBox_resolution = QComboBox(self.groupBox_9)
        self.comboBox_resolution.addItem("")
        self.comboBox_resolution.addItem("")
        self.comboBox_resolution.addItem("")
        self.comboBox_resolution.setObjectName(u"comboBox_resolution")
        self.comboBox_resolution.setMinimumSize(QSize(200, 50))
        self.comboBox_resolution.setFont(font3)

        self.gridLayout_10.addWidget(self.comboBox_resolution, 4, 2, 1, 1)

        self.pushButton_upload_i2v_image = QPushButton(self.groupBox_9)
        self.pushButton_upload_i2v_image.setObjectName(u"pushButton_upload_i2v_image")
        self.pushButton_upload_i2v_image.setMinimumSize(QSize(0, 50))
        self.pushButton_upload_i2v_image.setFont(font3)

        self.gridLayout_10.addWidget(self.pushButton_upload_i2v_image, 1, 1, 1, 2)

        self.textEdit_prompt = QTextEdit(self.groupBox_9)
        self.textEdit_prompt.setObjectName(u"textEdit_prompt")
        self.textEdit_prompt.setMinimumSize(QSize(0, 0))
        self.textEdit_prompt.setMaximumSize(QSize(16777215, 16777215))
        self.textEdit_prompt.setFont(font3)
        self.textEdit_prompt.setReadOnly(False)

        self.gridLayout_10.addWidget(self.textEdit_prompt, 3, 1, 1, 2)

        self.textEdit_download_status = QTextEdit(self.groupBox_9)
        self.textEdit_download_status.setObjectName(u"textEdit_download_status")
        self.textEdit_download_status.setMinimumSize(QSize(0, 0))
        self.textEdit_download_status.setMaximumSize(QSize(16777215, 40))
        self.textEdit_download_status.setFont(font3)
        self.textEdit_download_status.setReadOnly(True)

        self.gridLayout_10.addWidget(self.textEdit_download_status, 6, 0, 1, 3)

        self.lineEdit_duration = QLineEdit(self.groupBox_9)
        self.lineEdit_duration.setObjectName(u"lineEdit_duration")
        self.lineEdit_duration.setMinimumSize(QSize(200, 50))
        self.lineEdit_duration.setFont(font3)

        self.gridLayout_10.addWidget(self.lineEdit_duration, 4, 1, 1, 1)


        self.verticalLayout_14.addLayout(self.gridLayout_10)


        self.gridLayout_3.addWidget(self.groupBox_9, 4, 0, 1, 1)

        self.groupBox_8 = QGroupBox(self.scrollAreaWidgetContents_3)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.groupBox_8.setMinimumSize(QSize(500, 800))
        self.groupBox_8.setFont(font2)
        self.verticalLayout_20 = QVBoxLayout(self.groupBox_8)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.gridLayout_11 = QGridLayout()
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.frame_8 = QFrame(self.groupBox_8)
        self.frame_8.setObjectName(u"frame_8")
        sizePolicy1.setHeightForWidth(self.frame_8.sizePolicy().hasHeightForWidth())
        self.frame_8.setSizePolicy(sizePolicy1)
        self.frame_8.setMinimumSize(QSize(512, 512))
        self.frame_8.setStyleSheet(u"    border: 2px solid #555;       /* \u8fb9\u6846\u7c97\u7ec6 + \u989c\u8272 */\n"
"    border-radius: 5px;           /* \u5706\u89d2 */\n"
"    background-color: #1e1e1e;    /* \u80cc\u666f\u8272 */")
        self.frame_8.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_19 = QVBoxLayout(self.frame_8)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.graphicsView_image_sr_output = QGraphicsView(self.frame_8)
        self.graphicsView_image_sr_output.setObjectName(u"graphicsView_image_sr_output")

        self.verticalLayout_19.addWidget(self.graphicsView_image_sr_output)


        self.gridLayout_11.addWidget(self.frame_8, 1, 1, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.pushButton_upload_sr_image = QPushButton(self.groupBox_8)
        self.pushButton_upload_sr_image.setObjectName(u"pushButton_upload_sr_image")
        self.pushButton_upload_sr_image.setMinimumSize(QSize(0, 50))
        self.pushButton_upload_sr_image.setFont(font3)

        self.horizontalLayout_2.addWidget(self.pushButton_upload_sr_image)

        self.pushButton_image_sr = QPushButton(self.groupBox_8)
        self.pushButton_image_sr.setObjectName(u"pushButton_image_sr")
        self.pushButton_image_sr.setMinimumSize(QSize(0, 50))
        self.pushButton_image_sr.setFont(font3)

        self.horizontalLayout_2.addWidget(self.pushButton_image_sr)

        self.pushButton_export_sr_image = QPushButton(self.groupBox_8)
        self.pushButton_export_sr_image.setObjectName(u"pushButton_export_sr_image")
        self.pushButton_export_sr_image.setMinimumSize(QSize(0, 50))
        self.pushButton_export_sr_image.setFont(font3)

        self.horizontalLayout_2.addWidget(self.pushButton_export_sr_image)


        self.gridLayout_11.addLayout(self.horizontalLayout_2, 2, 0, 1, 2)

        self.frame_7 = QFrame(self.groupBox_8)
        self.frame_7.setObjectName(u"frame_7")
        sizePolicy1.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy1)
        self.frame_7.setMinimumSize(QSize(512, 512))
        self.frame_7.setStyleSheet(u"    border: 2px solid #555;       /* \u8fb9\u6846\u7c97\u7ec6 + \u989c\u8272 */\n"
"    border-radius: 5px;           /* \u5706\u89d2 */\n"
"    background-color: #1e1e1e;    /* \u80cc\u666f\u8272 */")
        self.frame_7.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_18 = QVBoxLayout(self.frame_7)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.graphicsView_image_sr_input = QGraphicsView(self.frame_7)
        self.graphicsView_image_sr_input.setObjectName(u"graphicsView_image_sr_input")

        self.verticalLayout_18.addWidget(self.graphicsView_image_sr_input)


        self.gridLayout_11.addWidget(self.frame_7, 1, 0, 1, 1)

        self.label_5 = QLabel(self.groupBox_8)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMinimumSize(QSize(0, 100))
        self.label_5.setFont(font3)

        self.gridLayout_11.addWidget(self.label_5, 0, 0, 1, 2)


        self.verticalLayout_20.addLayout(self.gridLayout_11)


        self.gridLayout_3.addWidget(self.groupBox_8, 3, 0, 1, 1)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_3)

        self.verticalLayout_10.addWidget(self.scrollArea_2)

        self.tabWidget.addTab(self.tab_2, "")

        self.verticalLayout_8.addWidget(self.tabWidget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1126, 33))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        MainWindow.setMenuBar(self.menubar)

        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.actionby_Asher)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u6566\u714c\u58c1\u753b\u56fe\u50cf\u4fee\u590d\u4e0e\u89c6\u89c9\u91cd\u5efa\u7cfb\u7edf", None))
        self.actionby_Asher.setText(QCoreApplication.translate("MainWindow", u"by_Asher", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"\u6566\u714c\u58c1\u753b\u56fe\u50cf\u4fee\u590d\u4e0e\u89c6\u89c9\u91cd\u5efa\u7cfb\u7edf", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("MainWindow", u"\u9996\u9875", None))
        self.groupBox_Seg.setTitle(QCoreApplication.translate("MainWindow", u"\u635f\u574f\u533a\u57df\u68c0\u6d4b\u6a21\u578b", None))
        self.pushButton_SAM_base_model_chose.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9SAM\u57fa\u5e95\u6a21\u578b\u6a21\u578b", None))
        self.pushButton_SAM_finetune_model_chose.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9SAM-Adapter\u5fae\u8c03\u6a21\u578b", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf\u4fee\u590d\u6a21\u578b", None))
        self.pushButton_sd_model_chose.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9StableDiffusion\u6a21\u578b", None))
        self.pushButton_controlnet_model_chose.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9ControlNet\u6a21\u578b", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"\u8d85\u5206\u8fa8\u7387\u6a21\u578b", None))
        self.pushButton_sr_model_chose.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9SR\u6a21\u578b", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"\u901a\u4e49\u4e07\u76f8API", None))
        self.pushButton_application.setText(QCoreApplication.translate("MainWindow", u"\u5e94\u7528", None))
        self.pushButton_clean.setText(QCoreApplication.translate("MainWindow", u"\u6e05\u7a7a", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"\u72b6\u6001\u4fe1\u606f", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"\u5bfc\u5165\u6a21\u578b", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"\u7ebf\u7a3f\u63d0\u53d6", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"1\u3001\u63d0\u53d6\u6216\u4e0a\u4f20\u7ebf\u7a3f\n"
"2\u3001\u81ea\u52a8\u6216\u624b\u52a8\u906e\u7f69\u635f\u574f\u533a\u57df\n"
"3\u3001\u5728\u9009\u5b9a\u7684\u906e\u7f69\u533a\u57df\u8fdb\u884c\u7ebf\u7a3f\u4fee\u8865", None))
        self.pushButton_pointer.setText(QCoreApplication.translate("MainWindow", u"\u6307\u9488", None))
        self.pushButton_line_art_extraction.setText(QCoreApplication.translate("MainWindow", u"\u63d0\u53d6\u7ebf\u7a3f", None))
        self.pushButton_paintbrush.setText(QCoreApplication.translate("MainWindow", u"\u753b\u7b14", None))
        self.pushButton_eraser.setText(QCoreApplication.translate("MainWindow", u"\u6a61\u76ae", None))
        self.pushButton_upload_line.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4f20\u7ebf\u7a3f", None))
        self.pushButton_auto_mask.setText(QCoreApplication.translate("MainWindow", u"\u81ea\u52a8\u906e\u7f69", None))
        self.pushButton_mask_paintbrush.setText(QCoreApplication.translate("MainWindow", u"\u906e\u7f69\u753b\u7b14", None))
        self.pushButton_mask_eraser.setText(QCoreApplication.translate("MainWindow", u"\u906e\u7f69\u6a61\u76ae", None))
        self.pushButton_pointer_2.setText(QCoreApplication.translate("MainWindow", u"\u6307\u9488", None))
        self.pushButton_all_mask.setText(QCoreApplication.translate("MainWindow", u"\u5168\u906e\u7f69", None))
        self.pushButton_clean_all_mask.setText(QCoreApplication.translate("MainWindow", u"\u6e05\u9664\u906e\u7f69", None))
        self.pushButton_export_line.setText(QCoreApplication.translate("MainWindow", u"\u7ebf\u7a3f\u5bfc\u51fa", None))
        self.pushButton_export_mask.setText(QCoreApplication.translate("MainWindow", u"\u906e\u7f69\u5bfc\u51fa", None))
        self.groupBox_7.setTitle(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf\u4fee\u590d", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"1\u3001\u5b8c\u6210\u7ebf\u7a3f\u548c\u906e\u7f69\u7684\u7ed8\u5236\n"
"2\u3001\u8bbe\u5b9a\u597d\u6269\u6563\u6b65\u6570\uff08\u6b63\u6574\u6570\uff09\n"
"3\u3001\u70b9\u51fb\u5f00\u59cb\u56fe\u50cf\u4fee\u590d\u6309\u94ae\u5e76\u7b49\u5f85\u4fee\u590d\u5b8c\u6210\n"
"4\u3001\u5bfc\u51fa\u4fee\u590d\u597d\u7684\u58c1\u753b\u56fe\u50cf", None))
        self.pushButton_export_restore_image.setText(QCoreApplication.translate("MainWindow", u"\u5bfc\u51fa\u4fee\u590d\u56fe\u50cf", None))
        self.lineEdit_step.setText("")
        self.lineEdit_step.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\u6269\u6563\u6b65\u6570", None))
        self.pushButton_image_restoration.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb\u56fe\u50cf\u4fee\u590d", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("MainWindow", u"\u635f\u574f\u533a\u57df\u68c0\u6d4b\u6a21\u5757", None))
        self.pushButton_upload_image.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4f20\u56fe\u7247", None))
        self.pushButton_seg_detect.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb\u635f\u574f\u533a\u57df\u68c0\u6d4b", None))
        self.lineEdit_device.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\u8fd0\u884c\u8bbe\u5907", None))
        self.lineEdit_size.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\u8f93\u5165\u56fe\u50cf\u5c3a\u5bf8", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"1\u3001\u4e0a\u4f20\u9700\u8981\u4fee\u590d\u7684\u6566\u714c\u58c1\u753b\u56fe\u50cf\u3002\n"
"2\u3001\u7cfb\u7edf\u5c06\u4f7f\u7528\u5bfc\u5165\u7684\u6a21\u578b\u8fdb\u884c\u635f\u574f\u533a\u57df\u68c0\u6d4b\u3002\n"
"3\u3001\u4e0b\u8f7d\u68c0\u6d4b\u7ed3\u679c\u3002", None))
        self.groupBox_9.setTitle(QCoreApplication.translate("MainWindow", u"\u89c6\u9891\u5316", None))
        self.comboBox_model.setItemText(0, QCoreApplication.translate("MainWindow", u"wan2.6-i2v-flash", None))
        self.comboBox_model.setItemText(1, QCoreApplication.translate("MainWindow", u"wan2.6-i2v", None))
        self.comboBox_model.setItemText(2, QCoreApplication.translate("MainWindow", u"wan2.5-i2v-preview", None))
        self.comboBox_model.setItemText(3, QCoreApplication.translate("MainWindow", u"wan2.2-i2v-flash", None))
        self.comboBox_model.setItemText(4, QCoreApplication.translate("MainWindow", u"wan2.2-i2v-plus", None))
        self.comboBox_model.setItemText(5, QCoreApplication.translate("MainWindow", u"wanx2.1-i2v-plus", None))
        self.comboBox_model.setItemText(6, QCoreApplication.translate("MainWindow", u"wanx2.1-i2v-turbo", None))

        self.label_4.setText(QCoreApplication.translate("MainWindow", u"1\u3001\u4e0a\u4f20\u4fee\u590d\u597d\u7684\u56fe\u50cf\n"
"2\u3001\u8bbe\u5b9a\u63d0\u793a\u8bcd\u3001\u65f6\u957f\u3001\u5206\u8fa8\u7387\u3001\u6a21\u578b\u7b49\u53c2\u6570\n"
"3\u3001\u70b9\u51fb\u52a8\u6001\u5316\u6309\u94ae\u5b8c\u6210\u56fe\u7247\u52a8\u6001\u5316\n"
"4\u3001\u5404\u6a21\u578b\u652f\u6301\u7684\u53c2\u6570\u4e0d\u4e00\u6837\uff0c\u8be6\u60c5\u8bf7\u5230\u901a\u4e49\u4e07\u76f8\u6a21\u578b\u8bf4\u660e\u6587\u6863\u67e5\u770b", None))
        self.pushButton_image_to_video.setText(QCoreApplication.translate("MainWindow", u"\u52a8\u6001\u5316", None))
        self.pushButton_play_pause.setText(QCoreApplication.translate("MainWindow", u"\u64ad\u653e/\u6682\u505c", None))
        self.comboBox_resolution.setItemText(0, QCoreApplication.translate("MainWindow", u"1080P", None))
        self.comboBox_resolution.setItemText(1, QCoreApplication.translate("MainWindow", u"720P", None))
        self.comboBox_resolution.setItemText(2, QCoreApplication.translate("MainWindow", u"480P", None))

        self.pushButton_upload_i2v_image.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4f20\u56fe\u50cf", None))
#if QT_CONFIG(tooltip)
        self.textEdit_prompt.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.textEdit_prompt.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\u8f93\u5165Prompt\uff08\u63d0\u793a\u8bcd\uff09", None))
#if QT_CONFIG(tooltip)
        self.textEdit_download_status.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.textEdit_download_status.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\u72b6\u6001\u4fe1\u606f", None))
        self.lineEdit_duration.setText("")
        self.lineEdit_duration.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\u89c6\u9891\u65f6\u957f\uff08\u79d2\uff09", None))
        self.groupBox_8.setTitle(QCoreApplication.translate("MainWindow", u"\u8d85\u5206\u8fa8\u7387", None))
        self.pushButton_upload_sr_image.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4f20\u56fe\u50cf", None))
        self.pushButton_image_sr.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb\u56fe\u50cf\u91cd\u5efa", None))
        self.pushButton_export_sr_image.setText(QCoreApplication.translate("MainWindow", u"\u5bfc\u51fa\u91cd\u5efa\u56fe\u50cf", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"1\u3001\u4e0a\u4f20\u4fee\u590d\u597d\u7684\u58c1\u753b\u56fe\u50cf\n"
"2\u3001\u70b9\u51fb\u5f00\u59cb\u56fe\u50cf\u8d85\u5206\u6309\u94ae\u5e76\u7b49\u5f85\u8d85\u5206\u5b8c\u6210\n"
"3\u3001\u5bfc\u51fa\u8d85\u5206\u8fa8\u7387\u7684\u58c1\u753b\u56fe\u50cf", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"\u6a21\u578b\u63a8\u7406", None))
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"\u4f5c\u8005", None))
    # retranslateUi

