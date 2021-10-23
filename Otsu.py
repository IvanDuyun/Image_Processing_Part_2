import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
import numpy as np

def otsu(gray_img):
    pixel_number = gray_img.shape[0] * gray_img.shape[1]
    mean_weigth = 1.0 / pixel_number
    # Вычисляем гистограмму изображения
    his, bins = np.histogram(gray_img, np.array(range(0, 255)))
    final_thresh = -1
    final_value = -1
    # Для каждого t в гистограмме
    for t in bins[1:-1]:
        # Считаем вероятности классов,разделенных порогом t
        W1 = np.sum(his[:t]) * mean_weigth
        # Считаем средние арифметические классов
        mu1 = np.mean(his[:t])
        mu2 = np.mean(his[t:])
        # Считаем дисперсию
        value = W1 * (1-W1)* (mu1 - mu2) ** 2
        # Если дисперсия больше чем имеющееся, то запоминаем новые значения
        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray_img.copy()
    final_img[gray_img > final_thresh] = 255
    final_img[gray_img < final_thresh] = 0
    return final_img


img = data.brick()
gray = rgb2gray(img)
otsu_img = otsu(gray)
fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(otsu_img, cmap=plt.cm.gray)
ax[1].set_title('Otsu')
plt.show()
