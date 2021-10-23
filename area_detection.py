from collections import defaultdict

from skimage import data
from skimage.util import img_as_ubyte
from matplotlib import pyplot as plt
import numpy as np

def area_detection(img):
    mask = np.zeros_like(img, dtype='int32')
    equals = defaultdict(set)
    num = 0

    # Первый проход
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Если пиксель не нулевой
            if img[i, j]:
                # Просмотр соседей
                top = mask[i-1, j] if i else mask[i, j]
                left = mask[i, j-1] if j else mask[i, j]
                # Если левый сосед не нулевой, присваиваем его значение
                if left:
                    mask[i, j] = left
                    # Если метки различаются, то сохраняем пару
                    if top and (left != top):
                        equals[top].add(left)
                        equals[left].add(top)
                # Если верхний сосед не нулевой, присваиваем его значение
                elif top:
                    mask[i, j] = top
                # Если все соседи нулевые присваивается новая неиспользованная метка
                else:
                    num += 1
                    mask[i, j] = num
                    equals[num].add(num)

    # Составляем все объединения множеств эквивалентностей
    for key in equals.keys():
        for elem in equals[key].copy():
            equals[elem].update(equals[key].copy())

    # Новая маска для второго прохода
    mask_new = np.zeros_like(img)

    # Второй проход
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                mask_new[i, j] = min(equals[mask[i, j]])

    unique_num = np.unique(mask_new)
    for i, num in enumerate(unique_num):
        mask_new[mask_new == num] = 255//len(unique_num) * i
    return mask_new


orig_img = img_as_ubyte(data.binary_blobs())
detect_img = area_detection(orig_img)

fig, axes = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
axes[0].imshow(orig_img, cmap='gray')
axes[1].imshow(detect_img)
plt.show()
