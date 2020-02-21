import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix


# ------------------------------------------------------------------
# Generar datos
# ------------------------------------------------------------------

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
print(np.shape(imagenes), n_imagenes)

data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ------------------------------------------------------------------
# Componentes Principales de los Unos
# ------------------------------------------------------------------

numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

# ------------------------------------------------------------------
# Criterio
# ------------------------------------------------------------------

def evaluar_uno(prueba_0):
    
    pc1,pc2,pc3 = np.dot(prueba_0,vectores[:,0]),np.dot(prueba_0,vectores[:,1]),np.dot(prueba_0,vectores[:,2])

    mu_dim1 = np.mean(np.matmul(x_train[ii],vectores[:,0]))
    mu_dim2 = np.mean(np.matmul(x_train[ii],vectores[:,1]))
    mu_dim3 = np.mean(np.matmul(x_train[ii],vectores[:,2]))

    sigma_dim1 = np.std(np.matmul(x_train[ii],vectores[:,0]))
    sigma_dim2 = np.std(np.matmul(x_train[ii],vectores[:,1]))
    sigma_dim3 = np.std(np.matmul(x_train[ii],vectores[:,2]))

    cond_1 = pc1<=(mu_dim1+sigma_dim1) and pc1>=(mu_dim1-sigma_dim1)
    cond_2 = pc2<=(mu_dim2+sigma_dim2) and pc2>=(mu_dim1-sigma_dim2)
    cond_3 = pc3<=(mu_dim3+sigma_dim3) and pc3>=(mu_dim1-sigma_dim3)

    return (cond_1 and cond_2 and cond_3)


# ------------------------------------------------------------------
# Solución
# ------------------------------------------------------------------

y_binario = y_test==1
predicciones = []

for i in x_test:
    predicciones.append(evaluar_uno(i))
    
matriz = confusion_matrix(y_binario,predicciones)
print(matriz)
P = matriz[0,0]/(matriz[1,0]+matriz[0,0])
R = matriz[0,0]/(matriz[0,1]+matriz[0,0])
F1 = 2*P*R/(P+R)

graficar = matriz*1
plt.imshow(graficar)
plt.title(r'F1 = {:.2f}'.format(F1))
plt.text(0,0,'TP = {:.2f}'.format(matriz[0,0]),fontsize=20)
plt.text(0,1,'FP = {:.2f}'.format(matriz[1,0]),fontsize=20)
plt.text(1,0,'FN = {:.2f}'.format(matriz[0,1]),fontsize=20)
plt.text(1,1,'TN = {:.2f}'.format(matriz[1,1]),fontsize=20)
plt.axis('off')
plt.savefig('matriz_de_confusion.png')