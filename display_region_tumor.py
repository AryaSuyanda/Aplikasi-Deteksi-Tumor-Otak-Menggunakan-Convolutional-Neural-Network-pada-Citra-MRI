import numpy as np
import cv2 as cv

from ImagePreprocessor import ImagePreprocessor

class DisplayTumor:
    def __init__(self, img):
        self.orig_img = np.array(img)
        self.cur_img = np.array(img)
        self.kernel = np.ones((3, 3), np.uint8)
        self.thresh = None

    def remove_noise(self):
        # Metode untuk menghilangkan noise dari gambar
        imagePre = ImagePreprocessor()  # Membuat objek dari class ImagePreprocessor
        gray = imagePre.toGrayScale(self.orig_img)  # Mengkonversi gambar ke grayscale
        ret, self.thresh = cv.threshold(gray, 0, 255,
                                        cv.THRESH_OTSU)  # Menerapkan threshold Otsu untuk mengubah gambar menjadi biner
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, self.kernel,
                                  iterations=2)  # Menggunakan operasi morfologi 'open' untuk menghapus noise
        self.cur_img = opening  # Menetapkan gambar saat ini setelah menghapus noise

    def display_tumor(self):
        # Metode untuk menampilkan tumor pada gambar
        if self.thresh is None:  # Jika gambar belum di threshold, jalankan fungsi remove_noise
            self.remove_noise()

        sure_bg = cv.dilate(self.cur_img, self.kernel,
                            iterations=3)  # Melakukan dilasi untuk mendapatkan area latar belakang yang pasti

        dist_transform = cv.distanceTransform(self.cur_img, cv.DIST_L2,
                                              5)  # Menghitung transformasi jarak untuk mendapatkan area foreground yang pasti
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255,
                                    0)  # Menerapkan threshold pada hasil transformasi jarak

        sure_fg = np.uint8(sure_fg)  # Mengubah hasil threshold menjadi uint8
        unknown = cv.subtract(sure_bg,
                              sure_fg)  # Menghitung area yang tidak diketahui dengan mengurangi area foreground dari background

        ret, markers = cv.connectedComponents(
            sure_fg)  # Membuat marker untuk komponen yang terhubung pada area foreground

        markers = markers + 1  # Menambahkan satu ke semua label agar background pasti tidak 0, tetapi 1

        markers[unknown == 255] = 0  # Menandai area yang tidak diketahui dengan nol
        markers = cv.watershed(self.orig_img,
                               markers)  # Melakukan segmentasi watershed pada gambar asli dengan marker yang telah dibuat
        self.orig_img[markers == -1] = [255, 0,
                                        0]  # Menandai batas-batas antara objek di gambar asli dengan warna biru

        tumor_image = cv.cvtColor(self.orig_img, cv.COLOR_HSV2BGR)  # Mengkonversi gambar dari ruang warna HSV ke BGR
        self.cur_img = tumor_image  # Menetapkan gambar saat ini menjadi gambar tumor

    def get_current_image(self):
        # Metode untuk mendapatkan gambar saat ini
        return self.cur_img  # Mengembalikan gambar saat ini
