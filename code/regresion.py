import numpy as np
import pandas as np
import matplotlib.pyplot as plt
import ipywidgets as wg
from ipywidgets import interactive, fixed

# Regression

df = pd.read_csv('data/media.csv');
print(df.shape);
df.head(10);

x = df.height;
y = df.Peso

plt.figure()
plt.scatter(x, y)
plt.xlabel('Altura')
plt.ylabel('Peso')

def plot_line(w, b):
    plt.figure(0, figsize=(20,4))
    plt.subplot(1,3,3)
    plt.scatter(x, y)
    y_pred = x*w + b
    plt.plot(x, y_pred, c='red')
    plt.xlim(140, 210)
    plt.ylim(40, 120)
    
    plt.subplot(1,3,2)
    x_ = np.array([0, x.max()])
    y_ = x_*w + b
    plt.scatter(x, y)
    plt.plot(x_, y_, c='red')
    plt.xlim(0, 210)
    plt.ylim(-160, 120)
    
    plt.subplot(1,3,1)
    mse = np.mean((y - y_pred)**2)
    loss.append(mse)
    plt.plot(loss)
    plt.title('Loss')
    
    plt.show()

loss = []

interactive_plot = interactive(plot_line, w=(1, 1.5, 0.01), b=(-200, 0, 1))
output = interactive_plot.children[-1]
output.layout_height = '350px'
interactive_plot

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x.values.reshape(-1,1), y)
print("w: {:.2f} \nb: {:.2f}".format(reg.coef_[0], reg.intercept_))