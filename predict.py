import cv2 as cv
import imutils
from keras.models import load_model

from ImagePreprocessor import ImagePreprocessor


class TumorDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)




    def preprocess_image(self, image):
        # Metode untuk melakukan pra-pemrosesan pada gambar
        imagePreprocessor = ImagePreprocessor()  # Membuat objek dari class ImagePreprocessor
        gray = imagePreprocessor.toGrayScale(image)  # Mengubah gambar menjadi grayscale
        blurred = imagePreprocessor.applyGaussianBlur(gray)  # Mengaplikasikan Gaussian Blur pada gambar grayscale
        thresh = imagePreprocessor.applyThresholding(blurred)  # Mengaplikasikan thresholding pada gambar yang telah di-blur
        erored = imagePreprocessor.applyErosian(thresh)  # Mengaplikasikan erosian pada gambar yang telah ditentukan threshold-nya
        dilated = imagePreprocessor.applyDilation(erored)  # Mengaplikasikan dilation pada gambar yang telah diberi erosian

        cnts = imagePreprocessor.findContours(dilated)  # Mencari kontur pada gambar yang telah diberi dilation
        extLeft, extRight, extTop, extBot = imagePreprocessor.findExtremePoints(cnts)  # Menemukan titik ekstrem pada kontur
        image = imagePreprocessor.cropAndResizeImage(image, extLeft, extRight, extTop, extBot)  # Memotong dan mengubah ukuran gambar berdasarkan titik ekstrem

        return image  # Mengembalikan gambar yang telah diproses

    def predict_tumor(self, image):
        # Metode untuk memprediksi keberadaan tumor pada gambar
        preprocessed_image = self.preprocess_image(image)  # Melakukan pra-pemrosesan pada gambar
        res = self.model.predict(preprocessed_image)  # Melakukan prediksi dengan model pada gambar yang telah diproses
        return res  # Mengembalikan hasil prediksi


