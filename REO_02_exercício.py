print("--------------------------------------Exercício 01----------------------------------------------------")

import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

print(" ")
print("a)Apresentea  imageme asinformações de  número  de  linhase  colunas; número  de  canaise número total de pixels")
print(" ")
trat17 = "17.jpg"
img_bgr = cv2.imread(trat17, 1)

lin, col, canais = np.shape(img_bgr)

print("Dimensão: " + str(lin) + " x " + str(col) + "; Número de Canais: " + str(canais) + "; Pixels: " + str(lin*col))
print(" ")

print("b)Faça um recorte da imagem para obter somente a área de interesse. Utilize esta imagem para a solução das próximas alternativas")
print(" ")

b, g, r = cv2.split(img_bgr)
img_rgb = cv2.merge([r, g ,b]) # Transformando a imagem BGR para RGB


plt.figure("Imagem")
plt.imshow(img_rgb)
plt.show()

#Recortando os canais em x = 0 até 2360 e y = 2630 até o final da imagem (Folha mais afetada)

rec_rgb = img_rgb[:2360,2630:,:]

plt.figure("Recorte")
plt.imshow(rec_rgb)
plt.show()

lin, col, canais = np.shape(rec_rgb)
rec_bgr = cv2.cvtColor(rec_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("T17_3.jpg",rec_bgr)

print("Dimensão: " + str(lin) + " x " + str(col) + "; Número de Canais: " + str(canais) + "; Pixels: " + str(lin*col))
print(" ")

print("c)Converta a imagem colorida para uma de escala de cinza (intensidade) e a apresente utilizando os mapas de cores “Escala de Cinza” e “JET”")
print(" ")
T17_3 = "T17_3.jpg"

img_gray = cv2.imread(T17_3, 0)

vet = np.array([0,25,50,75,100,125,150,175,200,225,255], np.uint8)
print(vet)
print(" ")
mat = np.ones((2,11))
print(" ")
print(mat)
print(" ")
mat = mat*vet
print(mat)

plt.figure("Intensidade")
plt.subplot(2,2,1)
plt.imshow(img_gray, cmap = "gray")
plt.title("Cinza")
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(img_gray, cmap = "jet")
plt.title("JET")
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(mat, cmap = "gray")
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(mat, cmap = "jet")
plt.xticks([])
plt.yticks([])

plt.show()

print(" ")
print("d) Apresente aimagemem escala de cinza e o seu respectivo histograma; Relacione o histograma e a imagem.")
print(" ")

hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0,256])

plt.figure("Histograma Cinza")
plt.subplot(2,1,1)
plt.imshow(img_gray, cmap = "gray")
plt.title("Imagem")
plt.xticks([])
plt.yticks([])

plt.subplot(2,1,2)
plt.plot(hist_gray, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()

print("e) Utilizando a imagem em escala de cinza (intensidade) realize a segmentação da imagem de modo a " +
"remover o fundo daimagem utilizando um limiar manual e o limiar obtido pela técnica de  Otsu. " +
"Nesta  questão  apresenteo  histograma  com marcação  dos  limiares  utilizados, " +
"a imagem limiarizada (binarizada) e a imagem colorida finalobtida da segmentação. Explique os resultados.")
print(" ")
limiar_manual = 150

(L1, img_limiar_manual) = cv2.threshold(img_gray,limiar_manual,255,cv2.THRESH_BINARY_INV)
(L2, img_limiar_otsu) = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

print("Limiar Manual: " + str(L1))
print("Limiar Otsu: " + str(L2))

plt.figure("Thresholding")
plt.subplot(3,3,2)
plt.imshow(img_gray, cmap = "gray")
plt.title("Imagem")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,4)
plt.imshow(img_limiar_manual, cmap = "gray")
plt.title("Limiarização Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,6)
plt.imshow(img_limiar_otsu, cmap = "gray")
plt.title("Limiarização de Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,7)
plt.plot(hist_gray, color = "black")
plt.title("Maual")
plt.axvline(x = L1, color = "r")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,9)
plt.plot(hist_gray, color = "black")
plt.title("Otsu")
plt.axvline(x = L2, color = "r")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()


rec_segmentada = cv2.bitwise_and(rec_rgb, rec_rgb, mask = img_limiar_otsu)

plt.imshow(rec_segmentada)
plt.xticks([])
plt.yticks([])

plt.show()

print(" ")

print("f)Apresente uma figura contendo a imagem selecionada nos sistemas RGB, Lab, HSVe YCrCb.")

print(" ")

rec_lab = cv2.cvtColor(rec_rgb, cv2.COLOR_RGB2Lab)
rec_hsv = cv2.cvtColor(rec_rgb, cv2.COLOR_RGB2HSV)
rec_ycrcb = cv2.cvtColor(rec_rgb, cv2.COLOR_RGB2YCR_CB)

plt.figure("Sistemas de Cores")
plt.subplot(2,2,1)
plt.imshow(rec_rgb)
plt.title("RGB")
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(rec_lab)
plt.title("Lab")
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(rec_hsv)
plt.title("HSV")
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(rec_ycrcb)
plt.title("YCRCB")
plt.xticks([])
plt.yticks([])

plt.show()



print("g)Apresente  uma  figura para  cadaum  dossistemasde cores(RGB,  HSV,  Lab e  YCrCb)contendo a imagem de cada um dos canais e seus respectivos histogramas.")

###Histogramas dos canais RGB####

hist_R = cv2.calcHist([rec_rgb], [0], None, [256], [0,256])
hist_G = cv2.calcHist([rec_rgb], [1], None, [256], [0,256])
hist_B = cv2.calcHist([rec_rgb], [2], None, [256], [0,256])

###Histogramas dos canais HSV####

hist_H = cv2.calcHist([rec_hsv], [0], None, [256], [0,256])
hist_S = cv2.calcHist([rec_hsv], [1], None, [256], [0,256])
hist_V = cv2.calcHist([rec_hsv], [2], None, [256], [0,256])

###Histogramas dos canais Lab####

hist_L = cv2.calcHist([rec_lab], [0], None, [256], [0,256])
hist_a = cv2.calcHist([rec_lab], [1], None, [256], [0,256])
hist_b = cv2.calcHist([rec_lab], [2], None, [256], [0,256])

###Histogramas dos canais YCRCB####

hist_Y = cv2.calcHist([rec_ycrcb], [0], None, [256], [0,256])
hist_CR = cv2.calcHist([rec_ycrcb], [1], None, [256], [0,256])
hist_CB = cv2.calcHist([rec_ycrcb], [2], None, [256], [0,256])

#########Histogramas e Canais#############

plt.figure("RGB e Histogramas")
plt.subplot(3,3,1)
plt.imshow(rec_rgb[:,:,0], cmap = "gray")
plt.title("R")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,2)
plt.imshow(rec_rgb[:,:,1], cmap = "gray")
plt.title("G")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,3)
plt.imshow(rec_rgb[:,:,2], cmap = "gray")
plt.title("B")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,4)
plt.plot(hist_R, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,5)
plt.plot(hist_G, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,6)
plt.plot(hist_B, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()


plt.figure("Lab e Histogramas")
plt.subplot(3,3,1)
plt.imshow(rec_lab[:,:,0], cmap = "gray")
plt.title("L")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,2)
plt.imshow(rec_lab[:,:,1], cmap = "gray")
plt.title("a")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,3)
plt.imshow(rec_lab[:,:,2], cmap = "gray")
plt.title("b")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,4)
plt.plot(hist_L, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,5)
plt.plot(hist_a, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,6)
plt.plot(hist_b, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()

plt.figure("HSV e Histogramas")
plt.subplot(3,3,1)
plt.imshow(rec_hsv[:,:,0], cmap = "gray")
plt.title("H")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,2)
plt.imshow(rec_hsv[:,:,1], cmap = "gray")
plt.title("S")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,3)
plt.imshow(rec_hsv[:,:,2], cmap = "gray")
plt.title("V")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,4)
plt.plot(hist_H, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,5)
plt.plot(hist_S, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,6)
plt.plot(hist_V, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()


plt.figure("YCRCB e Histogramas")
plt.subplot(3,3,1)
plt.imshow(rec_ycrcb[:,:,0], cmap = "gray")
plt.title("Y")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,2)
plt.imshow(rec_ycrcb[:,:,1], cmap = "gray")
plt.title("CR")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,3)
plt.imshow(rec_ycrcb[:,:,2], cmap = "gray")
plt.title("CB")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,4)
plt.plot(hist_Y, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,5)
plt.plot(hist_CR, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,6)
plt.plot(hist_CB, color = "black")
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()

print("h)Encontre o sistema de cor e o respectivo canal que propicie melhor segmentação da imagemde modo a remover o " +
"fundo da imagem utilizando limiar manual elimiar obtido pela técnica de Otsu. "+
"Nesta  questão  apresente o  histograma  com marcação  dos  limiares  utilizados, a " +
"imagem limiarizada (binarizada) e a imagem colorida finalobtida dasegmentação. " +
"Explique os resultados de sua  escolha pelo sistema de  cor  e canal utilizado na segmentação. " +
"Nesta questão  apresente a  imagem limiarizada  (binarizada)  e  a  imagem colorida finalobtida  dasegmentação.")

#O melhor Canal para individualizar a folha foi o S do sistema HSV.

limiar = 50
(L1, img_limiar_manual_S) = cv2.threshold(rec_hsv[:,:,1],limiar,255,cv2.THRESH_BINARY)
(L2, img_limiar_otsu_S) = cv2.threshold(rec_hsv[:,:,1],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

rec_seg_M = cv2.bitwise_and(rec_rgb, rec_rgb, mask = img_limiar_manual_S)
rec_seg_O = cv2.bitwise_and(rec_rgb, rec_rgb, mask = img_limiar_otsu_S)

plt.figure("Imagem Segmentada")
plt.subplot(3,2,1)
plt.imshow(img_limiar_manual_S, cmap = "gray")
plt.xticks([])
plt.yticks([])
plt.title("Segmentação Manual")

plt.subplot(3,2,2)
plt.imshow(img_limiar_otsu_S, cmap = "gray")
plt.xticks([])
plt.yticks([])
plt.title("Segmentação Otsu")

plt.subplot(3,2,5)
plt.plot(hist_S, color = "black")
plt.title("Limiar Manual")
plt.axvline(x = L1, color = "r")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,2,6)
plt.plot(hist_S, color = "black")
plt.title("Limiar Otsu")
plt.axvline(x = L2, color = "r")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,2,3)
plt.imshow(rec_seg_M)
plt.xticks([])
plt.yticks([])

plt.subplot(3,2,4)
plt.imshow(rec_seg_O)
plt.xticks([])
plt.yticks([])

plt.show()

print(" ")
print("h)Obtenha o histograma de cada um dos canais da imagem em RGB, utilizando como mascara a imagem limiarizada(binarizada) da letra h.")
print(" ")

hist_bin_R = cv2.calcHist([rec_rgb], [0], img_limiar_otsu_S, [256], [0,256])
hist_bin_G = cv2.calcHist([rec_rgb], [1], img_limiar_otsu_S, [256], [0,256])
hist_bin_B = cv2.calcHist([rec_rgb], [2], img_limiar_otsu_S, [256], [0,256])

plt.figure("Histogrma Mascara")
plt.subplot(3,1,1)
plt.plot(hist_bin_R, color = "r")
plt.title("R")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,1,2)
plt.plot(hist_bin_G, color = "g")
plt.title("G")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,1,3)
plt.plot(hist_bin_B, color = "b")
plt.title("B")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()


print("j)Realize operações aritméticas na imagemem RGB de  modo  a  realçar os aspectos  de seu interesse. " +
"Exemplo (2*R-0.5*G). Explique a sua escolha pelas operações aritméticas. Segue abaixo algumas sugestões.")
print(" ")



R, G, B = cv2.split(rec_seg_O)

rec_new = 128 + (112.439*R)/256 - (94.154*G)/256 - (18.285*B)/256 # CANAL CR do YCRCB

rec_new = rec_new.astype(np.uint8)

hist_new = cv2.calcHist([rec_new], [0], img_limiar_otsu_S, [256], [0,256])

(L1, rec_new_bin) = cv2.threshold(rec_new,130,255, cv2.THRESH_BINARY)

rec_seg_Fer = cv2.bitwise_and(rec_rgb, rec_rgb, mask = rec_new_bin)

plt.figure("Normalized Green Red Difference Index")
plt.subplot(1,3,1)
plt.imshow(rec_seg_O)

plt.subplot(1,3,2)
plt.imshow(rec_new_bin, cmap = "gray")

plt.subplot(1,3,3)
plt.imshow(rec_seg_Fer)

plt.show()











