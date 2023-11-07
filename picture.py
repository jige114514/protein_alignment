import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

pid = ['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7']
not_trained = [0.53301663, 0.68271771, 0.79970777, 0.86583871, 0.92718792, 0.93936514]
baseline = [0.55403043, 0.7631658, 0.8640777, 0.89485455, 0.96322796, 0.98103517]
trained = [0.73469388, 0.85785016, 0.92642352, 0.95401599, 0.96774194, 0.99269006]
unsupervised = [0.7008761, 0.85536093, 0.90609133, 0.9457649, 0.96671165, 0.98976608]  # batch=3条非同源
supervised = [0.742586, 0.87390407, 0.93087188, 0.95447734, 0.96685021, 0.99269006]
supervised2 = [0.73469388, 0.86928442, 0.93041829, 0.95515807, 0.96797393, 0.99269006]

plt.plot(pid, not_trained, label='not trained', marker='o')
plt.plot(pid, baseline, label='baseline', marker='o')
plt.plot(pid, trained, label='trained', marker='o')
plt.plot(pid, unsupervised, label='unsupervised', marker='o')
plt.plot(pid, supervised, label='supervised', marker='o')
plt.plot(pid, supervised2, label='supervised2', marker='o')
plt.xlabel('Percent identity')
plt.ylabel('F1')
plt.legend()
plt.grid()
plt.show()
