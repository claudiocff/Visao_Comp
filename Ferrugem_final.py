####Carregar pacotes
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

os.chdir('D:\My Drive\Arquivos-notebook\Python-projects\Ferrugem\Imagens')

path_of_images='D:\My Drive\Arquivos-notebook\Python-projects\Ferrugem\Imagens\Fotos_Tocha'####direterio das imagens
mask_folhas ='D:\My Drive\Arquivos-notebook\Python-projects\Ferrugem\Imagens\mask_folha_tocha'
mask_lesoes ='D:\My Drive\Arquivos-notebook\Python-projects\Ferrugem\Imagens\lesoes_tocha'
filenames= glob.glob(path_of_images + "/*.jpg")
dimen = []
i=1

for imagem in filenames:
    img = cv2.imread(imagem)
    # img = img[300:-300, 300:-300] # Para fotos Tocha usar essa linha
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    s = cv2.medianBlur(s, 35)
    _, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    folhas_s_fundo = cv2.bitwise_and(img_rgb.copy(), img_rgb.copy(), mask=thresh)
    nome_legenda = os.path.basename(imagem)
    cv2.imwrite(os.path.join(mask_folhas, nome_legenda), folhas_s_fundo)

    ####OBTENDO CADA FOLHA EM SEPARADO###
    mask = np.zeros(img_rgb.shape, dtype=np.uint8)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    ####Obtendo as lesoes
    lesoes = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCR_CB)
    h, se, v = cv2.split(lesoes)
    _,thrsh1 = cv2.threshold(se, 135, 255, cv2.THRESH_BINARY)
    lesoes1 = cv2.bitwise_and(img_rgb, img_rgb, mask=thrsh1)
    cv2.imwrite(os.path.join(mask_lesoes, nome_legenda), lesoes1)

    a = i / (len(filenames)) * 100
    print("{0:.2f} % completed".format(round(a, 2)))
    i = i + 1

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        obj = folhas_s_fundo[y:y + h, x:x + w]
        obj_bgr = cv2.cvtColor(obj, cv2.COLOR_RGB2BGR)
        area = cv2.contourArea(c)
        razao = (h / w).__round__(2)
        ####Lesoes
        les = thrsh1[y:y + h, x:x + w]
        area_lesao = cv2.countNonZero(les)
        contorno = cv2.findContours(les, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorno = contorno[0] if len(contorno) == 2 else contorno[1]
        contorno = len(contorno)
        if area_lesao == 0:
            razao_lesao = 0
        else:
            razao_lesao = ((area_lesao / area) * 100).__round__(2)
        nome_leg = nome_legenda[:-4]
        dimen += [[str(nome_leg),str(i + 1), str(h), str(w), str(area), str(razao),
                   str(area_lesao), str(contorno), str(razao_lesao),'CLAUDIO']]


dados_folhas = pd.DataFrame(dimen)
dados_folhas = dados_folhas.rename(columns={0:'Imagem',1:'FOLHA', 2: 'ALTURA_FOLHA',3:'LARGURA_FOLHA',4:'AREA_FOLHA',5:'RAZAO_FOLHA',
                                            6:'AREA_LESÃO',7:'NUMERO DE PUSTULAS',8:'RAZAO DA LESÃO',9:'NOME'})
dados_folhas.to_csv('medidas.csv',index=False)