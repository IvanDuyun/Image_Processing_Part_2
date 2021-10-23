import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float
from task1 import otsu_img


def dilation(orig_img, cernel_s):
    img = img_as_float(orig_img)
    dilation_img = img.copy()
    for i in range(2, img.shape[0] - 2):
        for j in range(2, img.shape[1] - 2):
            # Перемножаем ядро и часть пикселей
            a = cernel_s * img[(i - 1):(i + 2), (j - 1):(j + 2)]
            # Если получилась матрица единиц, то расширяем область до 5*5 с единицами
            if np.array_equal(a, cernel_s):
                dilation_img[(i - 2):(i + 3), (j - 2):(j + 3)] = 1

    return dilation_img


def erosion(orig_img, cernel_s):
    img = img_as_float(orig_img)
    erosion_img = img.copy()
    # Для эрозии неоьходим поиск минимума, поэтому матрицу умножаем на 0
    cernel = cernel_s * 0
    for i in range(2, img.shape[0] - 2):
        for j in range(2, img.shape[1] - 2):
            # Складываем ядро и часть пикселей
            a = cernel + img[(i - 1):(i + 2), (j - 1):(j + 2)]
            # Если получилась матрица нулей, то расширяем область до 5*5 с нулями
            if np.array_equal(a, cernel):
                erosion_img[(i - 2):(i + 3), (j - 2):(j + 3)] = 0
    return erosion_img


# Ядро 3*3
cernel_s = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]])

orig_img = otsu_img
dilated = dilation(orig_img, cernel_s)
eroded = erosion(orig_img, cernel_s)

fig, (ax, ax1, ax2) = plt.subplots(ncols=3)
ax.imshow(orig_img, cmap=plt.cm.gray)
ax.set_title('Otsu')
ax1.imshow(dilated, cmap=plt.cm.gray)
ax1.set_title('Dilation')
ax2.imshow(eroded, cmap=plt.cm.gray)
ax2.set_title('Erosion')
plt.show()
