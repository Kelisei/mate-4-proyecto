# REGRESION LINEAL MULTIPLE

import pandas as pd
from sympy import symbols, sqrt
from scipy.stats import t
import numpy as np
df= pd.read_csv('winequality-red.csv', delimiter=';')

y = df['quality']
X = df.drop(columns=['quality'])
variables_predictoras = df[['volatile acidity', 'density', 'sulphates', 'alcohol', 'citric acid']]
normalized = (variables_predictoras - variables_predictoras.mean()) / variables_predictoras.std()

# Agregar columna de 1s para el intercepto
X_b = np.c_[np.ones((normalized.shape[0], 1)), normalized]

# Inicialización
m, n = X_b.shape
beta_grad = np.zeros(n)
alpha = 0.1
epochs = 1000
min_err = 1e-6
# Descenso del gradiente
for epoch in range(epochs):
    y_pred = X_b.dot(beta_grad)
    error = y_pred - y
    grad = (1/m) * X_b.T.dot(error)
    beta_grad -= alpha * grad

    # Verificar si el error es menor que el mínimo permitido
    if np.linalg.norm(grad, ord=1) < min_err:
        print(f"Convergió en la época {epoch}")
        break

print("Coeficientes estimados:", beta_grad)
 


# Cálculo de los coeficientes usando minimos cuadrados

# Añadimos columna de 1s para el intercepto
Z_b = np.c_[np.ones(normalized.shape[0]), normalized]
# Coeficientes de regresión
beta_cuad = np.linalg.inv(Z_b.T.dot(Z_b)).dot(Z_b.T).dot(y)
print("Coeficientes estimados:", beta_cuad)


y_pred_cuad = Z_b.dot(beta_cuad)
SSr = ((y - y_pred_cuad)**2).sum()
SSt = ((y - y.mean())**2).sum()
R2 = 1 - SSr/SSt
print("R^2:", R2)
k = 5
Ra2 = 1 - (1-R2)*(len(y)-k)/(len(y) - k - 1)
print("R^2 ajustado:", Ra2)
ra = sqrt(Ra2)
print("r (correlación):", ra)





























































































































































































































































#                              ...,?77??!~~~~!???77?<~.... 
#                         ..?7`                           `7!.. 
#                     .,=`          ..~7^`   I                  ?1. 
#        ........  ..^            ?`  ..?7!1 .               ...??7 
#       .        .7`        .,777.. .I.    . .!          .,7! 
#       ..     .?         .^      .l   ?i. . .`       .,^ 
#        b    .!        .= .?7???7~.     .>r .      .= 
#        .,.?4         , .^         1        `     4... 
#         J   ^         ,            5       `         ?<. 
#        .%.7;         .`     .,     .;                   .=. 
#        .+^ .,       .%      MML     F       .,             ?, 
#         P   ,,      J      .MMN     F        6               4. 
#         l    d,    ,       .MMM!   .t        ..               ,, 
#         ,    JMa..`         MMM`   .         .!                .; 
#          r   .M#            .M#   .%  .      .~                 ., 
#        dMMMNJ..!                 .P7!  .>    .         .         ,, 
#        .WMMMMMm  ?^..       ..,?! ..    ..   ,  Z7`        `?^..  ,, 
#           ?THB3       ?77?!        .Yr  .   .!   ?,              ?^C 
#             ?,                   .,^.` .%  .^      5. 
#               7,          .....?7     .^  ,`        ?. 
#                 `<.                 .= .`'           1 
#                 ....dn... ... ...,7..J=!7,           ., 
#              ..=     G.,7  ..,o..  .?    J.           F 
#            .J.  .^ ,,,t  ,^        ?^.  .^  `?~.      F 
#           r %J. $    5r J             ,r.1      .=.  .% 
#           r .77=?4.    ``,     l ., 1  .. <.       4., 
#           .$..    .X..   .n..  ., J. r .`  J.       `' 
#         .?`  .5        `` .%   .% .' L.'    t 
#         ,. ..1JL          .,   J .$.?`      . 
#                 1.          .=` ` .J7??7<.. .; 
#                  JS..    ..^      L        7.: 
#                    `> ..       J.  4. 
#                     +   r `t   r ~=..G. 
#                     =   $  ,.  J 
#                     2   r   t  .; 
#               .,7!  r   t`7~..  j.. 
#               j   7~L...$=.?7r   r ;?1. 
#                8.      .=    j ..,^   .. 
#               r        G              . 
#             .,7,        j,           .>=. 
#          .J??,  `T....... %             .. 
#       ..^     <.  ~.    ,.             .D 
#     .?`        1   L     .7.........?Ti..l 
#    ,`           L  .    .%    .`!       `j, 
#  .^             .  ..   .`   .^  .?7!?7+. 1 
# .`              .  .`..`7.  .^  ,`      .i.; 
# .7<..........~<<3?7!`    4. r  `          G% 
#                           J.` .!           % 
#                             JiJ           .` 
#                               .1.         J 
#                                  ?1.     .'         
#                                      7<..%


