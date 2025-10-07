import pandas as pd
from sympy import symbols, sqrt
from scipy.stats import t
df= pd.read_csv('winequality-red.csv', delimiter=';')

y = df['quality']
X = df.drop(columns=['quality'])
variables_predictoras = df[['volatile acidity', 'density', 'sulphates', 'alcohol', 'citric acid']]

alpha = 0.05
n = len(y)
print(f'n: {n}')
v = n - 2
t_critico = t.ppf(1-alpha/2, v)
print(f't_critico: {t_critico}\n')

for col in variables_predictoras.columns:
    x = symbols('x', real=True)
    x_promedio = variables_predictoras[col].mean()
    y_promedio = y.mean()
    Sxx = ((variables_predictoras[col] - x_promedio) ** 2).sum()
    Sxy = ((variables_predictoras[col] - x_promedio) * (y - y_promedio)).sum()
    Syy = ((y - y_promedio) ** 2).sum()
    b1 = Sxy / Sxx
    b0 = y_promedio - b1 * x_promedio
    SSR = Syy - b1 * Sxy
    y_estimado = b1*x + b0
    varianza_estimada = SSR / (n - 2)
    R2 = 1 - (SSR / Syy)
    r = sqrt(R2)
    e_b0 = t_critico * sqrt(varianza_estimada) * (1/n + (x_promedio**2 / Sxx))
    IC_b0_superior = b0 + e_b0
    IC_b0_inferior = b0 - e_b0
    e_b1 = t_critico * sqrt(varianza_estimada / Sxx)
    IC_b1_superior = b1 + e_b1
    IC_b1_inferior = b1 - e_b1
    e_media = t_critico * sqrt(varianza_estimada * (1/n + ((x_promedio - x_promedio)**2 / Sxx))) #chequear
    IC_media_superior = b0 + b1*x_promedio + e_media #chequear
    IC_media_inferior = b0 + b1*x_promedio - e_media #chequear
    e_pred = t_critico * sqrt(varianza_estimada * (1 + 1/n + ((x_promedio - x_promedio)**2 / Sxx))) #chequear
    IC_pred_superior = y_promedio + e_pred  #chequear
    IC_pred_inferior = y_promedio + b1*x_promedio - e_pred  #chequear
    print(f'Variable predictora: {col}')
    print(f'b0 (intercepto): {b0}')
    print(f'b1 (pendiente): {b1}')
    print(f'Varianza estimada: {varianza_estimada}')
    print(f'R²: {R2}')
    print(f'r (correlación): {r}\n')
    print(f'y_estimado: {y_estimado}\n')
    print(f'IC b0: ({IC_b0_inferior}, {IC_b0_superior})')
    print(f'IC b1: ({IC_b1_inferior}, {IC_b1_superior})')
    print(f'IC media para x={x_promedio}: ({IC_media_inferior   }, {IC_media_superior})')
    print(f'IC predicción para x={x_promedio}: ({IC_pred_inferior}, {IC_pred_superior})\n')
    print('---------------------------------\n')
    
    print("Alcohol")



