import cv2
from matplotlib import pyplot as plt
import skimage
import numpy as np
from skimage.measure import label, regionprops

img_bgr = cv2.imread("17.jpg", 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.figure("imgem")
plt.imshow(img_rgb)
plt.show()

#a) Aplique o filtro de média com cincodiferentes tamanhos de kernel e 
#compare os resultados com a imagem original;

img_fm_1 = cv2.blur(img_rgb,(11,11))
img_fm_2 = cv2.blur(img_rgb,(21,21))
img_fm_3 = cv2.blur(img_rgb,(31,31))
img_fm_4 = cv2.blur(img_rgb,(41,41))
img_fm_5 = cv2.blur(img_rgb,(51,51))

filtros = [img_rgb, img_fm_1, img_fm_2, img_fm_3, img_fm_4, img_fm_5]
titulos = ["Imagem Original", "Kernel 11 x 11", "Kernel 21 x 21", "Kernel 31 x 31", "Kernel 41 x 41","Kernel 51 x 51"]

for i in range(6):
    plt.subplot(2,3,i+1);plt.imshow(filtros[i])
    plt.xticks([]);plt.yticks([])
    plt.title(titulos[i])
plt.show()

# b) Aplique  diferentes  tipos  de  filtros compelomenos  dois  tamanhos  de  kernel e 
# compare  os resultados entre si e com a imagem original.


mediana_31 = cv2.medianBlur(img_rgb, 31) # Melhor para tirar o Ruído sem perder definição
mediana_51 = cv2.medianBlur(img_rgb, 51) 

gaussiano_31 = cv2.GaussianBlur(img_rgb,(31,31),0)
gaussiano_51 = cv2.GaussianBlur(img_rgb,(51,51),0)

bilateral_31 = cv2.bilateralFilter(img_rgb, 31, 31, 21)
bilateral_51 = cv2.bilateralFilter(img_rgb, 51, 51, 21)

filtros = [img_rgb, mediana_31, mediana_51, bilateral_31, bilateral_51, gaussiano_31, gaussiano_51]

titulos = ["Imagem Original", "Mediana 31 x 31", "Mediana 51 x 51", "Bilateral 31 x 31", "Bilateral 51 x 51", "Gaussiano 31 x 31", "Gaussiano 51 x 51"]

for i in range(7):
    plt.subplot(3,3,i+1);plt.imshow(filtros[i])
    plt.xticks([]);plt.yticks([])
    plt.title(titulos[i])
plt.show()

# c) Realize  a  segmentação  da imagem utilizando  o  processo  de  limiarização.  Utilizando  
# o reconhecimento de contornos, identifiquee salve os objetos de interesse. 
# Além disso, acesse as bibliotecas Opencv  e  Scikit-Image,  verifique  as variáveis  que  podem  ser 
# mensuradas e extraia as informações pertinentes (crie e salve uma tabela com estes dados).
# Apresente todas as imagens obtidas ao longo deste processo.

img_hsv = cv2.cvtColor(mediana_31, cv2.COLOR_RGB2HSV)
img_s = img_hsv[:,:,1]

plt.imshow(img_s, cmap = "jet")
plt.show()

(L1, img_bin) = cv2.threshold(img_s, 45, 255, cv2.THRESH_BINARY)

plt.imshow(img_bin, cmap = "jet")
plt.show()

img_seg = cv2.bitwise_and(img_rgb, img_rgb, mask = img_bin)

plt.imshow(img_seg , cmap = "jet")
plt.show()


mascara = img_bin.copy()
cnts, h = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    obj = img_bin[y:y+h, x:x + w]
    obj_rgb = img_seg[y:y + h, x:x + w]
    obj_bgr = cv2.cvtColor(obj_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("folha" + str(i + 1) + ".jpg", obj_bgr)
    cv2.imwrite("folha_bin" + str(i + 1) + ".jpg", obj)
    regiao = regionprops(obj)
    print("folha: " , str(i + 1))
    print("Dimensão da imagem: " , np.shape(obj))
    print("Medidas Físicas")
    print("Centroide: " , regiao[0].centroid)
    print("Comprimento do eixo menor: " , regiao[0].minor_axis_length)
    print("Comprimento do eixo maior: " , regiao[0].major_axis_length)
    print("Razão: " , regiao[0].major_axis_length/regiao[0].minor_axis_length)
    area = cv2.contourArea(c)

    print("Área: " , area)
    print("Perímetro: " , cv2.arcLength(c, True))
    print("Medidas de Cor")
    min_val_r, max_val_r, min_loc_r, max_loc_r = cv2.minMaxLoc(obj_rgb[:, :, 0], mask = obj)
    print("Valor mínimo no R: " , min_val_r, " - Posição: ", min_loc_r)
    print("Valor máximo no R: " , max_val_r, " - Posição: ", max_loc_r)
    med_val_r = cv2.mean(obj_rgb[:, :, 0], mask = obj)
    print("Média no Vermelho: ", med_val_r)

    min_val_g, max_val_g, min_loc_g, max_loc_g = cv2.minMaxLoc(obj_rgb[:, :, 1], mask = obj)
    print("Valor mínimo no G: " , min_val_g, " - Posição: ", min_loc_g)
    print("Valor máximo no G: " , max_val_g, " - Posição: ", max_loc_g)
    med_val_g = cv2.mean(obj_rgb[:, :, 1], mask = obj)
    print("Média no Verde: ", med_val_g)

    min_val_b, max_val_b, min_loc_b, max_loc_b = cv2.minMaxLoc(obj_rgb[:, :, 2], mask = obj)
    print("Valor mínimo no B: " , min_val_b, " - Posição: ", min_loc_b)
    print("Valor máximo no B: " , max_val_b, " - Posição: ", max_loc_b)
    med_val_b = cv2.mean(obj_rgb[:, :, 2], mask = obj)
    print("Média no Azul: ", med_val_b)

img_17_1 = cv2.imread("folha1.jpg",1)
img_17_2 = cv2.imread("folha2.jpg",1)
img_17_3 = cv2.imread("folha3.jpg",1)

f1 = 1426484.5; f2 = 1472995.5; f3 = 1632589.5

area_total = [f1, f2, f3]

folhas = [img_17_1,img_17_2,img_17_3]

for i in range(3):
    folhas[i] = cv2.cvtColor(folhas[i],cv2.COLOR_BGR2YCR_CB)

img_ycrcb_1 = folhas[0]
img_ycrcb_2 = folhas[1]
img_ycrcb_3 = folhas[2]

img_cr_1 = img_ycrcb_1[:,:,1]
img_cr_2 = img_ycrcb_2[:,:,1]
img_cr_3 = img_ycrcb_3[:,:,1]

folhas_cr = [img_cr_1, img_cr_2, img_cr_3]

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(folhas_cr[i], cmap = "jet")
plt.show()



Limiar = [0,0,0]
img_folha_bin = [0,0,0]

for i in range(3):
    (Limiar[i], img_folha_bin[i]) = cv2.threshold(folhas_cr[i], 133, 255, cv2.THRESH_BINARY)


for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(img_folha_bin[i], cmap = "gray")
plt.show()

area_lesao = []

for i in range(3):
    a = cv2.countNonZero(img_folha_bin[i])
    area_lesao.append(a)



print(area_total)
print(area_lesao)

area_total_arr = np.zeros(3)
area_lesao_arr = np.zeros(3)

for i in range(3):
    area_total_arr[i] = np.array(area_total[i])

for i in range(3):
    area_lesao_arr[i] = np.array(area_lesao[i])

print(area_lesao_arr)


prop = (area_lesao_arr/area_total_arr)*100
print(prop)

data = np.concatenate((np.matrix(area_total_arr),np.matrix(area_lesao_arr),np.matrix(prop)), axis = 0)
print(data)

np.savetxt('trat_17.txt', data, delimiter=' ')

# d)Utilizando máscaras,apresente o histograma somente dos objetos de interesse.

folhas = [img_17_1,img_17_2,img_17_3]

folhas_bin = [cv2.imread("folha_bin1.jpg",0),
              cv2.imread("folha_bin2.jpg",0),
              cv2.imread("folha_bin3.jpg",0)]


hist = []
j = 0
for i in range(3):
    a = cv2.calcHist([folhas[i]], [1], folhas_bin[j], [256], [0,256])
    hist.append(a)
    j = j + 1

hist[2]

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.plot(hist[i])
plt.show()


#e) Realize  a  segmentação da  imagem  utilizando  a  técnica  de  k-means. 
# Apresente  as  imagens obtidas neste processo.


pixels = gaussiano_31.reshape((-1,3))
valores=np.float32(pixels)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2)
k = 2

dist,labels,(centers) = cv2.kmeans(valores,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
val,contagens = np.unique(labels,return_counts=True)
centers = np.uint8(centers)
matriz_segmentada = centers[labels]
matriz_segmentada = matriz_segmentada.reshape(img_rgb.shape)

plt.imshow(matriz_segmentada)
plt.show()

#f) Realize a segmentação da imagem utilizando a técnica de watershed. 
# Apresente as imagens obtidas neste processo.

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage


plt.imshow(img_seg)
plt.show()

img_dist = ndimage.distance_transform_edt(img_bin)

plt.imshow(img_dist)
plt.show()

max_local = peak_local_max(img_dist, indices=False, min_distance = 300, labels = img_bin)

marcadores, n_marcadores = ndimage.label(max_local, structure=np.ones((3,3)))

print(np.unique(marcadores, return_counts = True))

img_ws = watershed(-img_dist, marcadores, mask = img_bin)

plt.imshow(img_ws)
plt.show()


imagens = [img_rgb, img_seg, img_bin, img_dist, img_ws]
titulos = ["Original", "Segmentada", "Binarizada", "Distancias", "Watershed"]

for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(imagens[i])
    plt.title(titulos[i])
    plt.xticks([])
    plt.yticks([])
plt.show()


