import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fig, ax = plt.subplots(1, 3)

# Latin
ax[0].text(0.5, 0.5, 'a', fontfamily='DejaVu Sans')

# Greek
ax[1].text(0.5, 0.5, 'Ω', fontfamily='DejaVu Sans')

# Korean
prop = fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
ax[2].text(0.5, 0.5, '가', fontproperties=prop)

fig.savefig('test_fonts2.png')
