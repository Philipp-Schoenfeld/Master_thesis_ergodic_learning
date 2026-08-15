import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.text(0.5, 0.5, '가', fontfamily='sans-serif')
fig.savefig('test_korean.png')
