import math
import sys
from datetime import datetime

from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QRadioButton, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import cv2
from keras.models import load_model
import cv2 as cv
import imutils
import numpy as np
import os
from ImagePreprocessor import ImagePreprocessor

from display_region_tumor import DisplayTumor
from predict import TumorDetector
from PyQt5 import QtCore, QtWidgets
model_path = 'best_model.h5'

from PIL import Image
class Gui(QMainWindow):
    def __init__(self):
        super(Gui, self).__init__()
        self.thresh = None
        loadUi('gui.ui', self)
        self.Image = None
        self.ImageResult = None
        self.target_size = (224, 224)
        self.pushButton.clicked.connect(self.browseWindow)
        self.predict_button.clicked.connect(self.check)
        self.actionTransform_to_grayscale.triggered.connect(self.stepGrayscale)
        self.actionapply_gaussian.triggered.connect(self.stepGaussianBlur)
        self.actionapply_thresholding.triggered.connect(self.stepThresholding)
        self.actionapply_eroded.triggered.connect(self.stepErosion)
        self.actionapply_dilated.triggered.connect(self.stepDilation)
        self.actionfind_contours.triggered.connect(self.stepFindContours)
        self.actionTransform_to_grayscale_2.triggered.connect(self.stepGrayscale)
        self.actionApply_tresholding.triggered.connect(self.stepFindAreaTresholding)
        self.actionApply_morphology.triggered.connect(self.stepFindAreaApplyMorhology)
        self.actionApply_dilate.triggered.connect(self.stepFindAreanApplyDilate)
        self.actiondinf_foreground_area.triggered.connect(self.stepFindForeGrondArea)
        self.actionfind_uknown_region.triggered.connect(self.stepFindUnknowArea)
        self.actionmark_the_region_of_unknown_with_zero.triggered.connect(self.stepFindMarkRegion)





        self.listOfWinFrame = []

    def browseWindow(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.jpg *.png *.jpeg)')
        self.Image = cv2.imread(file_name)
        self.displayImage()

    def displayImage(self , index = 0 ):
        height, width, channel = self.Image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.Image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_original.setPixmap(pixmap)
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label.setScaledContents(True)



    def readImage(self):
        self.listOfWinFrame = []
        self.listOfWinFrame.append(self)
        self.label_result.setText("")
        self.predictTumor()

    def predictTumor(self):
        # Fungsi ini dipanggil untuk memprediksi tumor
        print('trigger')  # Mencetak pesan "trigger"
        tumor_detector = TumorDetector(model_path)  # Membuat objek detector tumor
        result = tumor_detector.predict_tumor(self.Image)  # Memprediksi apakah ada tumor dalam gambar
        if result > 0.5:
            # Jika hasil prediksi lebih dari 0.5, berarti model memprediksi ada tumor
            self.label_result.setText("Brain has Tumor, Tumor Detected!")  # Menampilkan pesan ini pada label hasil
            self.label_result.setStyleSheet("color: red")  # Mengubah warna teks menjadi merah
            output_type = 'tumor'  # Menentukan jenis output sebagai 'tumor'
        else:
            # Jika hasil prediksi kurang dari atau sama dengan 0.5, berarti model memprediksi tidak ada tumor
            self.label_result.setText("Brain is healthy, No Tumor Detected")  # Menampilkan pesan ini pada label hasil
            self.label_result.setStyleSheet("color: green")  # Mengubah warna teks menjadi hijau
            output_type = 'no_tumor'  # Menentukan jenis output sebagai 'no_tumor'

        self.exportImage(output_type)  # Memanggil fungsi exportImage dengan output_type sebagai argumen
        self.displayTumor()  # Memanggil fungsi displayTumor untuk menampilkan gambar tumor

    def exportImage(self, output_type):
        # Fungsi ini digunakan untuk mengekspor gambar
        current_date = datetime.now().strftime('%Y%m%d')  # Mendapatkan tanggal saat ini dalam format 'YYYYMMDD'

        filename = f'{output_type}_{current_date}.jpg'  # Menamai file gambar dengan output_type dan tanggal saat ini
        output_path = os.path.join(os.getcwd(), output_type, filename)  # Menentukan path tempat gambar akan disimpan
        cv2.imwrite(output_path, self.Image)  # Menyimpan gambar di lokasi tersebut

        print(f"Image exported to: {current_date}")  # Mencetak pesan ini yang berisi tanggal saat gambar diekspor

    def removeNoise(self):
        # Fungsi ini digunakan untuk mengaktifkan tombol view dan menghapus teks pada label hasil
        self.listOfWinFrame[0].button_view.setEnabled(True)
        self.label_result.setText("")

    def displayTumor(self):
        # Fungsi ini digunakan untuk menampilkan gambar tumor
        display_tumor = DisplayTumor(self.Image)  # Membuat objek DisplayTumor
        display_tumor.remove_noise()  # Memanggil fungsi remove_noise pada objek tersebut untuk menghapus noise pada gambar
        display_tumor.display_tumor()  # Memanggil fungsi display_tumor pada objek tersebut untuk menampilkan gambar tumor

        tumor_image = display_tumor.get_current_image()  # Mendapatkan gambar tumor saat ini
        height, width, channel = tumor_image.shape  # Mendapatkan dimensi gambar (tinggi, lebar, dan saluran)
        bytes_per_line = 3 * width  # Menghitung jumlah byte per baris pada gambar
        q_image = QImage(tumor_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # Membuat QImage dari data gambar tumor. QImage digunakan untuk menampilkan gambar di GUI Qt.

        pixmap = QPixmap.fromImage(q_image)
        # Mengkonversi QImage ke QPixmap. QPixmap biasanya lebih efisien dibanding QImage untuk menampilkan gambar di GUI.

        self.predict_image.setPixmap(pixmap)
        # Menampilkan gambar pixmap di widget (predict_image).

        self.predict_image.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        # Menyetel penjajaran gambar ke tengah secara horizontal dan vertikal.

        self.predict_image.setScaledContents(True)
        # Menyetel skala konten menjadi True sehingga gambar dapat diskalakan untuk mengisi widget.

    def check(self):
        self.readImage()
        # Memanggil fungsi readImage() yang belum ditentukan dalam kode yang diberikan.

    def stepGrayscale(self):
        # Fungsi ini melakukan tahap pengolahan gambar menjadi skala abu-abu.
        imagePreprocessor = ImagePreprocessor()  # Membuat objek ImagePreprocessor.
        gray = imagePreprocessor.toGrayScale(self.Image)  # Mengubah gambar ke skala abu-abu.
        cv2.imshow('gray', gray)  # Menampilkan gambar skala abu-abu.
        cv2.waitKey(0)  # Menahan jendela terbuka sampai tombol ditekan.
        return gray  # Mengembalikan gambar skala abu-abu.

    def stepGaussianBlur(self):
        # Fungsi ini menerapkan Gaussian blur ke gambar.
        imagePreprocessor = ImagePreprocessor()  # Membuat objek ImagePreprocessor.
        gray = imagePreprocessor.toGrayScale(self.Image)  # Mengubah gambar ke skala abu-abu.
        blurred = imagePreprocessor.applyGaussianBlur(gray)  # Menerapkan Gaussian blur ke gambar skala abu-abu.
        cv2.imshow('blurred', blurred)  # Menampilkan gambar yang telah diberikan blur.
        cv2.waitKey(0)  # Menahan jendela terbuka sampai tombol ditekan.
        return blurred  # Mengembalikan gambar yang telah diberikan blur.

    def stepThresholding(self):
        # Fungsi ini menerapkan thresholding ke gambar.
        imagePreprocessor = ImagePreprocessor()  # Membuat objek ImagePreprocessor.
        gray = imagePreprocessor.toGrayScale(self.Image)  # Mengubah gambar ke skala abu-abu.
        blurred = imagePreprocessor.applyGaussianBlur(gray)  # Menerapkan Gaussian blur ke gambar skala abu-abu.
        thresh = imagePreprocessor.applyThresholding(blurred, 45, 255)  # Menerapkan thresholding ke gambar.
        cv2.imshow('thresh', thresh)  # Menampilkan gambar yang telah diberikan thresholding.
        cv2.waitKey(0)  # Menahan jendela terbuka sampai tombol ditekan.
        return thresh  # Mengembalikan gambar yang telah diberikan thresholding.

    def stepErosion(self):
        # Fungsi ini menerapkan operasi erosi ke gambar.
        imagePreprocessor = ImagePreprocessor()  # Membuat objek ImagePreprocessor.
        gray = imagePreprocessor.toGrayScale(self.Image)  # Mengubah gambar ke skala abu-abu.
        blurred = imagePreprocessor.applyGaussianBlur(gray)  # Menerapkan Gaussian blur ke gambar skala abu-abu.
        thresh = imagePreprocessor.applyThresholding(blurred, 45, 255)  # Menerapkan thresholding ke gambar.
        eroded = imagePreprocessor.applyErosian(thresh)  # Menerapkan operasi erosi ke gambar.
        cv2.imshow('eroded', eroded)  # Menampilkan gambar yang telah di-erosi.
        cv2.waitKey(0)  # Menahan jendela terbuka sampai tombol ditekan.
        return eroded  # Mengembalikan gambar yang telah di-erosi.

    def stepDilation(self):
        # Fungsi ini menerapkan operasi dilasi ke gambar.
        imagePreprocessor = ImagePreprocessor()  # Membuat objek ImagePreprocessor.
        gray = imagePreprocessor.toGrayScale(self.Image)  # Mengubah gambar ke skala abu-abu.
        blurred = imagePreprocessor.applyGaussianBlur(gray)  # Menerapkan Gaussian blur ke gambar skala abu-abu.
        thresh = imagePreprocessor.applyThresholding(blurred, 45, 255)  # Menerapkan thresholding ke gambar.
        eroded = imagePreprocessor.applyErosian(thresh)  # Menerapkan operasi erosi ke gambar.
        dilated = imagePreprocessor.applyDilation(eroded)  # Menerapkan operasi dilasi ke gambar.
        cv2.imshow('dilated', dilated)  # Menampilkan gambar yang telah di-dilasi.
        cv2.waitKey(0)  # Menahan jendela terbuka sampai tombol ditekan.
        return dilated  # Mengembalikan gambar yang telah di-dilasi.

    def stepFindContours(self):
        # Membuat objek ImagePreprocessor
        imagePreprocessor = ImagePreprocessor()

        # Konversi gambar ke grayscale
        gray = imagePreprocessor.toGrayScale(self.Image)

        # Mengaplikasikan Gaussian blur ke gambar grayscale
        blurred = imagePreprocessor.applyGaussianBlur(gray)

        # Mengaplikasikan thresholding ke gambar yang telah di-blur
        thresh = imagePreprocessor.applyThresholding(blurred, 45, 255)

        # Mengaplikasikan erosi ke gambar hasil thresholding
        eroded = imagePreprocessor.applyErosian(thresh)

        # Mengaplikasikan dilasi ke gambar hasil erosi
        dilated = imagePreprocessor.applyDilation(eroded)

        # Mencari kontur pada gambar hasil dilasi
        contours = imagePreprocessor.findContours(dilated)

        # Mencari titik ekstrem dari kontur
        extLeft, extRight, extTop, extBot = imagePreprocessor.findExtremePoints(contours)

        # Memotong dan mengubah ukuran gambar berdasarkan titik ekstrem
        image = imagePreprocessor.cropAndResizeImage(self.Image, extLeft, extRight, extTop, extBot)

        # Mengembalikan bentuk gambar ke original
        image = image.reshape(image.shape[1], image.shape[2], image.shape[3])

        # Mengalikan gambar dengan 255 untuk konversi kembali ke rentang piksel asli
        image = image * 255

        # Mengkonversi gambar ke numpy array
        image_array = np.array(image)

        # Mengubah numpy array ke PIL image
        image = Image.fromarray(image_array.astype('uint8'))

        # Menampilkan gambar
        image.show()

        return image

    def stepFindAreaTresholding(self):
        # Membuat objek ImagePreprocessor
        imagePreprocessor = ImagePreprocessor()

        # Konversi gambar ke grayscale
        gray = imagePreprocessor.toGrayScale(self.Image)

        # Mengaplikasikan thresholding pada gambar grayscale dengan metode Otsu
        tresh = imagePreprocessor.applyThresholding(gray, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Menampilkan gambar hasil thresholding
        cv2.imshow('area', tresh)
        cv2.waitKey(0)

        return tresh

    def stepFindAreaApplyMorhology(self):
        # Membuat objek ImagePreprocessor
        imagePre = ImagePreprocessor()

        # Konversi gambar ke grayscale
        gray = imagePre.toGrayScale(self.Image)

        # Mengaplikasikan thresholding pada gambar grayscale dengan metode Otsu
        ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Mengaplikasikan operasi morfologi opening pada gambar hasil thresholding
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

        # Menampilkan gambar hasil operasi morfologi opening
        cv2.imshow('opening', opening)

    def stepFindAreanApplyDilate(self):
        # Membuat objek ImagePreprocessor
        imagePre = ImagePreprocessor()

        # Konversi gambar ke grayscale
        gray = imagePre.toGrayScale(self.Image)

        # Mengaplikasikan thresholding pada gambar grayscale dengan metode Otsu
        ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Mengaplikasikan operasi morfologi opening pada gambar hasil thresholding
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

        # Menerapkan dilasi pada gambar hasil operasi morfologi opening
        sure_bg = cv.dilate(opening, np.ones((3, 3), np.uint8), iterations=3)

        # Menampilkan gambar hasil dilasi
        cv2.imshow('sure_bg', sure_bg)

    def stepFindForeGrondArea(self):

        imagePre = ImagePreprocessor()
        gray = imagePre.toGrayScale(self.Image)
        ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        sure_bg = cv.dilate(opening, np.ones((3, 3), np.uint8), iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # open in cv2
        cv2.imshow('dist_transform', sure_fg)

    def stepFindUnknowArea(self):

        imagePre = ImagePreprocessor()
        gray = imagePre.toGrayScale(self.Image)
        ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        sure_bg = cv.dilate(opening, np.ones((3, 3), np.uint8), iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # open in cv2
        cv2.imshow('unknown', unknown)

    def stepFindMarkRegion(self):

        imagePre = ImagePreprocessor()
        gray = imagePre.toGrayScale(self.Image)
        ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        sure_bg = cv.dilate(opening, np.ones((3, 3), np.uint8), iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1

        # Now mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv.watershed(self.Image, markers)
        self.Image[markers == -1] = [255, 0, 0]

        image = cv.cvtColor(self.Image, cv.COLOR_HSV2BGR)

        # open in cv2
        cv2.imshow('image', image)

app = QApplication([])
window = Gui()
window.setWindowTitle('Tumor detector')
window.show()
app.exec_()
