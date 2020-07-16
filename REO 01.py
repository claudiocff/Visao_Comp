########################################################################################################################
# DATA: 17/07/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# DISCENTE: Cláudio Carlos Fernandes Filho
# E-MAIL: claudiocff3@yahoo.com.br
# GITHUB: claudiocff
########################################################################################################################

# REO 01 - LISTA DE EXERCÍCIOS

import numpy as np

print("-----------------------------------------------EXERCÍCIO 01------------------------------------------")
print(" ")
print("a) Declare os valores 43.5,150.30,17,28,35,79,20,99.07,15 como um array numpy.")
print(" ")
arr = np.array([43.5, 
150.30, 
17.0, 
28.0, 
35.0, 
79.0, 
20.0, 
99.07, 
15.0
])
print ("Vector: " + str(arr))
print(" ")

print("b) Obtenha as informações de dimensão, média, máximo, mínimo e variância deste vetor.")
print(" ")
dimv = len(arr)
meanv = int(np.mean(arr))
maxv = np.max(arr)
minv = np.min(arr)
varv = int(np.var(arr))

info = np.array([
"Dimension: " + str(dimv),
"Mean: " + str(meanv),
"Max: " + str(maxv),
"Min: " + str(minv),
"Variance: " + str(varv)
])

print(info)
print(" ")
print("c) Obtenha um novo vetor em que cada elemento é dado pelo quadrado da diferença entre cada elemento do vetor declaradona letra a e o valor da média deste.")
print(" ")

new_arr = arr - meanv
print(new_arr**2)
print(" ")

print("d) Obtenha um novo vetor que contenha todos os valores superiores a 30.")
print(" ")

print("Original Vector: " + str(arr))
print("Vector > 30: " + str(arr[(arr > 30)]))#if TRUE print(x)
print(" ")

print("e) Identifique quais as posições do vetor original possuem valores superiores a 30.")
print(" ")

print("Original Vector: " + str(arr))
print("Positions > 30: " + str(np.where(arr > 30)))
print(" ")

print("f) Apresente um vetor que contenha os valores da primeira, quinta e última posição.")
print(" ")

print("Original Vector: " + str(arr))
print(np.array([
"First Position: " + str(arr[0]),
"Fith Position: " + str(arr[4]),
"Last Position: " + str(arr[len(arr)-1])
]))
print(" ")

print("g) Crie uma estrutura de repetição usando o for para apresentar cada valor e a sua respectiva posição durante as iterações.")
print(" ")

for i in range(dimv):
    j = i+1
    print (
        "Position: " + str(j) + (" - Value: " + str(arr[i]))
        )

print(" ")

print("h) Crie uma estrutura de repetição usando o for para fazer a soma dos quadrados de cada valor do vetor.")
print(" ")

q = np.zeros(dimv) #creating a vector with zeros
sq = 0
for i in range(dimv):
    q[i] = (arr[i])**2 #each squared value
    sq = sum(q) #sum of elements inside the vector 


print ("Squared Value: " + str(q))
print ("Sum of squares: " + str (sq))
print(" ")

print("i) Crie uma estrutura de repetição usando o while para apresentar todos os valores do vetor")
print(" ")

print(arr)
j = 0
while arr[j] > 1:
    print("Position: " + str(j) + " - Value: " + str(arr[j]))
    j = j + 1
    if j == (len(arr)):
       break
print(" ")

print("j) Crie um sequência de valores com mesmo tamanho do vetor original e que inicie em 1 e o passo seja também 1.")
print(" ")
new_arr = np.arange(1, len(arr) + 1, 1)
print(new_arr)
print(" ")

print("k) Concatene o vetor da letra a com o vetor da letra j.")
print(" ")

cvector = np.concatenate((arr, new_arr))
print(cvector)
print(" ")
print("-----------------------------------------------EXERCÍCIO 02------------------------------------------")
print(" ")

print("a) Declare a matriz abaixo com a biblioteca numpy.")
print(" ")
print("1 3 22")
print("2 8 18")
print("3 4 22")
print("4 1 23")
print("5 2 52")
print("6 2 18")
print("7 2 25")
print(" ")
mat = np.matrix([[1, 3 , 22],
                [2, 8, 18],
                [3, 4, 22],
                [4, 1, 23],
                [5, 2, 52],
                [6, 2, 18],
                [7, 2, 25]])
print(mat)
print(" ")

print("b) Obtenha o número de linhas e de colunas desta matriz")
print(" ")

dim_mat = np.shape(mat)
print("N_Row = " + str(dim_mat[0]) + ", N_Col = " + str(dim_mat[1]))
print(" ")

print("c) Obtenha as médias das colunas 2 e 3.")
print(" ")

print("Col 2 Mean: " + str(round(np.mean(mat[:,1]))))
print("Col 3 Mean: " + str(round(np.mean(mat[:,2]))))
print(" ")

print("d) Obtenha as médias das linhas considerando somente as colunas 2 e 3")
print(" ")

mat23 = mat[:,1:]
#print(mat23)
row_means = np.mean(mat23,axis = 1)
print(row_means)
print(" ")

print("e) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade " +
 "de uma doença e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade inferior a 5.")
print(" ")

bol_5 = np.asarray(mat[:,1]) < 5
gen = np.asarray(mat[:,0])
print("Genotypes lower than 5: " + str((gen[bol_5])))
print(" ")


print("f) Considerando que a primeira coluna seja a identificação de genótipos, " +
"a segunda nota de severidade de uma doença e a terceira peso de 100 grãos." +  
"Obtenha os genótipos que possuem peso de 100 grãos superior ou igual a 22.")
print(" ")

bol_22 = np.asarray(mat[:,2]) >= 22
gen = np.asarray(mat[:,0])
print("Genotypes greater than 22: " + str(gen[bol_22]))

print("g) Considerando que a primeira coluna seja a identificação de genótipos, " +
"a segunda nota de severidade de uma doença ee a terceira peso de 100 grãos. " +
"Obtenha os genótipos que possuem nota de severidade igual ou inferior a 3 e peso de 100 " + 
"grãos igual ou superior a 22.")
print(" ")

bol_joint = ((mat[:,2] >= 22) & (mat[:,1] <= 3))
#print(bol_joint)

print("Selected genotypes: " + str(gen[bol_joint]))

print("h) Crie uma estrutura de repetição com uso do for (loop) para apresentar na " +
"tela cada uma das posições da matriz e o seu respectivo valor. Utilize um iterador para mostrar " +
"ao usuário quantas vezes está sendo repetido. Apresente a seguinte mensagem a cada iteração 7 "+
"Na linha X e na coluna Y ocorre o valor: Z.Nesta estrutura crie uma lista que armazene os " +
"genótipos com peso de 100 grãos igual ou superior a 25")
print(" ")

count = 0
gen = []

for i in np.arange(0,dim_mat[0],1):
    if mat[i, 2] >= 25:
        gen.append(mat[i, 0])
    for j in np.arange(0,dim_mat[1],1):
        count += 1
        print('Iteraction: '+ str(count))
        print('Row ' + str(i) + ' Col ' + str(j) + ' Value: ' + str(mat[int(i),int(j)]))
        print('-'*50)
print ("Genotypes greater than 25: ")
print (gen)
print(" ")

print("--------------------------------------------------EXERCÍCIO 3-----------------------------------------")
print(" ")

print("a) Crie uma função em um arquivo externo (outro arquivo .py) " +
"para calcular a média e a variância amostral um vetor qualquer, baseada em um loop (for).")
print(" ")

print("As funções estão no arquivo Function 3a.py")
print(" ")

print("Simule três arrays com a biblioteca numpy de 10, 100, e 1000 valores e com distribuição normal "+
"com média 100 e variância 2500. Pesquise na documentação do numpy por funções de simulação.")
print(" ")
import math
n1 = 10
n2 = 100
n3 = 1000
mu = 100
sigma = math.sqrt(2500)
print("Mean =" + str(mu) + ", SD = " + str(sigma))
print(" ")
x = np.random.normal(mu, sigma, n1)
y = np.random.normal(mu, sigma, n2)
z = np.random.normal(mu, sigma, n3)
print(x)
print(" ")
print(y)
print(" ")
print(z)
print("")

print("c) Utilize a função criada na letra a para obter as médias e " +
"variâncias dos vetores simulados na letra b.")
print(" ")

from Function_3a import mean
from Function_3a import var

print("Means: ")
print(mean(x))
print(mean(y))
print(mean(z))
print(" ")
print("Variances: ")
print(var(x))
print(var(y))
print(var(z))
print(" ")

print("d) Crie histogramas com a biblioteca matplotlib dos vetores simulados com valores de 10, 100, 1000 e 100000.")
n4 = 100000
w = np.random.normal(mu, sigma, n4)

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
fig, axs = plt.subplots(1, tight_layout=True)
N, bins, patches = axs.hist(x, bins=5)
fracs = N / N.max()

norm = colors.Normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.title("N = 10")
plt.xlabel(" ")
plt.ylabel('Count')


fig, axs = plt.subplots(1, tight_layout=True)
N, bins, patches = axs.hist(y, bins=10)
fracs = N / N.max()

norm = colors.Normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.title("N = 100")
plt.xlabel(" ")
plt.ylabel("Count")


fig, axs = plt.subplots(1, tight_layout=True)
N, bins, patches = axs.hist(z, bins=15)
fracs = N / N.max()

norm = colors.Normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.title("N = 1000")
plt.xlabel(" ")
plt.ylabel("Count")


fig, axs = plt.subplots(1, tight_layout=True)

N, bins, patches = axs.hist(w, bins=70)

fracs = N / N.max()

norm = colors.Normalize(fracs.min(), fracs.max())


for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.title('N = 100000')
plt.xlabel(' ')
plt.ylabel('Count')
plt.show()

#############################################TESTE PLOTLY############################################
#import plotly.graph_objects as go
#import plotly.io as pio
#from plotly.subplots import make_subplots
#pio.renderers.default = "browser"
#import pandas as pd

#fig = make_subplots(rows = 2, cols = 2)

#trace0 = go.Histogram(x=x, histnorm='density')
#trace1 = go.Histogram(x=y, histnorm='density')
#trace2 = go.Histogram(x=z, histnorm='density')
#trace3 = go.Histogram(x=w, histnorm='density')

#fig.append_trace(trace0, 1, 1)
#fig.append_trace(trace1, 1, 2)
#fig.append_trace(trace2, 2, 1)
#fig.append_trace(trace3, 2, 2)

#fig.show()

print(" ")

print("-------------------------------------------------EXERCÍCIO 4--------------------------------------------")
print(" ")
print("a) O arquivo dados.txt contem a avaliação de genótipos (primeira coluna) em " +
"repetições (segunda coluna) quanto a quatro ariáveis (terceira coluna em diante). Portanto, "+
"carregue o arquivo dados.txt com a biblioteca numpy, apresente os dados e obtenha as informações "+
"de dimensão desta matriz.")
print(" ")

import numpy as np

url = 'https://raw.githubusercontent.com/VQCarneiro/Visao-Computacional-no-Melhoramento-de-Plantas/master/Roteiros%20de%20Estudo%20Orientado%20-%20REOs/REO%2001/EXERC%C3%8DCIOS/dados.txt'

df = np.loadtxt(url)

print(df)
print(" ")
print(np.shape(df))
print(" ")

print("b) Pesquise sobre as funções np.unique e np.where da biblioteca numpy")
print(" ")

#help(np.unique)
#help(np.where)
print(" ")

print("c) Obtenha de forma automática os genótipos e quantas repetições foram avaliadas")
print(" ")

print('Genotypes: ')
gen = np.unique(df[0:30,0:1], axis=0)
nrow,ncol = np.shape(gen)

print(np.unique(df[0:30,0:1], axis=0))
print('REP: ')
print(np.unique(df[0:30,1:2], axis=0))
print(" ")

print("d) Apresente uma matriz contendo somente as colunas 1, 2 e 4")
print(" ")
df2 = df[:,[0,1,3]]
print(df2)
print(" ")

print(" e) Obtenha uma matriz que contenha o máximo, o mínimo, " + 
"a média e a variância de cada genótipo para a variavel da coluna 4. " +
"Salve esta matriz em bloco de notas.")
print(" ")

mingen = np.zeros((np.shape(gen)[0],1))
maxgen = np.zeros((np.shape(gen)[0],1))
meangen = np.zeros((np.shape(gen)[0],1))
vargen = np.zeros((np.shape(gen)[0],1))
it = 0
col4 = np.asarray(df[:,2])

for i in np.arange(0,np.shape(df2)[0],3):
    mingen[it,0] = np.min(df2[i:i + 3, 2], axis=0)
    maxgen[it,0] = np.max(df2[i:i + 3, 2], axis=0)
    meangen[it,0] = np.mean(df2[i:i + 3, 2], axis=0)
    vargen[it,0] = np.var(df2[i:i + 3, 2], axis=0)
    it = it + 1

cmat = np.concatenate((gen,mingen,maxgen,meangen,vargen),axis=1)
print(cmat)
np.savetxt('matriz_ex4.txt', cmat, delimiter=' ')
print(" ")

print("f) Obtenha os genótipos que possuem média (médias das repetições) " + 
"igual ou superior a 500 da matriz gerada na letra anterior.")
print(" ")

df3 = np.loadtxt('matriz_ex4.txt')
gen500 = np.squeeze(np.asarray(df3[:,3])) >= 500
print(gen[gen500])
print(" ")

print("g) Apresente os seguintes graficos: Médias dos genótipos para cada variável. " + 
"Utilizar o comando plt.subplot para mostrar mais de um grafico por figura")
print(" ")

from matplotlib import pyplot as plt

mean1 = np.zeros((np.shape(gen)[0],1))
mean2 = np.zeros((np.shape(gen)[0],1))
mean3 = np.zeros((np.shape(gen)[0],1))
mean4 = np.zeros((np.shape(gen)[0],1))
mean5 = np.zeros((np.shape(gen)[0],1))
it=0

for i in np.arange(0,np.shape(df)[0],3):
    mean1[it,0] = np.mean(df[i:i + 3, 2], axis=0)
    mean2[it,0] = np.mean(df[i:i + 3, 3], axis=0)
    mean3[it,0] = np.mean(df[i:i + 3, 4], axis=0)
    mean4[it,0] = np.mean(df[i:i + 3, 5], axis=0)
    mean5[it,0] = np.mean(df[i:i + 3, 6], axis=0)
    it = it + 1
df_mean = np.concatenate((gen,
mean1,
mean2,
mean3,
mean4,
mean5),
axis=1)

nr,nc = np.shape(df_mean)

plt.style.use('ggplot')
plt.figure('Mean Plot')
plt.subplot(2,3,1)
plt.bar(df_mean[:,0],df_mean[:,1])
plt.title('V 1')
plt.xticks(df_mean[:,0])
plt.subplot(2,3,2)
plt.bar(df_mean[:,0],df_mean[:,2])
plt.title('V 2')
plt.xticks(df_mean[:,0])
plt.subplot(2,3,3)
plt.bar(df_mean[:,0],df_mean[:,3])
plt.title('V 3')
plt.xticks(df_mean[:,0])
plt.subplot(2,3,4)
plt.bar(df_mean[:,0],df_mean[:,4])
plt.title('V 4')
plt.xticks(df_mean[:,0])
plt.subplot(2,3,5)
plt.bar(df_mean[:,0],df_mean[:,5])
plt.title('V 5')
plt.xticks(df_mean[:,0])
plt.show()
