import cv2 as cv
import numpy as np
import imutils
from keras.models import load_model

class ImagePreprocessor:
    def __init__(self, target_size=(240, 240)):
        # Inisiasi kelas dengan ukuran target untuk gambar yang akan diubah ukurannya
        self.target_size = target_size

    def toGrayScale(self, image):
        # Metode untuk mengubah gambar menjadi skala abu-abu (grayscale)
        h, w = image.shape[:2]
        gray = np.zeros((h, w), np.uint8)
        for i in range(h):
            for j in range(w):
                gray[i, j] = np.clip(0.3333 * image[i, j, 0] + 0.3333 * image[i, j, 1] + 0.3333 * image[i, j, 2], 0, 255)
        return gray


    def applyGaussianBlur(self, image):
        # kernel yang akan digunakan
        #kernel = (1.0 / 345) * np.array([[1, 5, 7, 5, 1],
            #                             [5, 20, 33, 20, 5],
           #                              [7, 33, 55, 33, 7],
          #                               [5, 20, 33, 20, 5],
         #                                [1, 5, 7, 5, 1]])

        #blurred = self.fungsi_konvolusi(image, kernel)
        blurred = cv.GaussianBlur(image, (5, 5), 0)  # Menerapkan Gaussian blur dengan kernel size 5x5
        return blurred  # Mengembalikan gambar yang sudah diblur

    def applyThresholding(self,image, lower=0 , upper=255  ,default=cv.THRESH_BINARY):
        # Metode untuk menerapkan thresholding ke gambar
        _, thresh = cv.threshold(image, lower, upper,default)  # Menerapkan thresholding ke gambar
        return thresh  # Mengembalikan gambar yang sudah di-threshold

    #def applyAdaptiveThresholding(self, image):
        # Metode untuk menerapkan adaptive thresholding ke gambar
       # thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        # Menerapkan adaptive thresholding ke gambar dengan method Gaussian C
       # return thresh  # Mengembalikan gambar yang sudah di-threshold secara adaptif

    def applyErosian(self, image):
        # Metode untuk menerapkan erosi ke gambar
        eroded = cv.erode(image, None, iterations=2)  # Menerapkan erosi ke gambar dengan 2 iterasi
        return eroded  # Mengembalikan gambar yang sudah di-erode

    def applyDilation(self, image):
        # Metode untuk menerapkan dilatasi ke gambar
        dilated = cv.dilate(image, None, iterations=2)  # Menerapkan dilatasi ke gambar dengan 2 iterasi
        return dilated  # Mengembalikan gambar yang sudah di-dilate

    def findContours(self, image):
        # Metode untuk menemukan kontur pada gambar
        cnts = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Mencari kontur pada gambar
        cnts = imutils.grab_contours(cnts)  # Mengambil kontur yang telah ditemukan
        return cnts  # Mengembalikan kontur yang ditemukan

    def findExtremePoints(self, contours):
        # Metode untuk menemukan titik ekstrem pada kontur
        c = max(contours, key=cv.contourArea)  # Mencari kontur dengan area terbesar
        # Menemukan titik ekstrem pada kontur
        extLeft = tuple(c[c[:, :, 0].argmin()][0])  # Titik ekstrem kiri
        extRight = tuple(c[c[:, :, 0].argmax()][0])  # Titik ekstrem kanan
        extTop = tuple(c[c[:, :, 1].argmin()][0])  # Titik ekstrem atas
        extBot = tuple(c[c[:, :, 1].argmax()][0])  # Titik ekstrem bawah
        return extLeft, extRight, extTop, extBot  # Mengembalikan titik ekstrem

    def cropAndResizeImage(self, image, extLeft, extRight, extTop, extBot):
        # Metode untuk memotong dan mengubah ukuran gambar
        cropped_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]  # Memotong gambar berdasarkan titik ekstrem
        # Mengubah ukuran gambar ke target size
        resized_image = cv.resize(cropped_image, dsize=self.target_size, interpolation=cv.INTER_CUBIC)
        normalized_image = resized_image / 255.  # Normalisasi pixel gambar ke range 0-1
        # Mengubah bentuk gambar untuk ditambahkan dimensi batch
        reshaped_image = normalized_image.reshape((1, *self.target_size, 3))
        return reshaped_image  # Mengembalikan gambar yang sudah dipotong, diubah ukurannya, dinormalisasi, dan diubah bentuknya
