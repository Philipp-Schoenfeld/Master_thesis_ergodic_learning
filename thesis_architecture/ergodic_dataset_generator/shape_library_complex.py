import numpy as np

COMPLEX_BUILDERS = {}
TEST_COMPLEX_SHAPES = []

def _register(name):
    def decorator(fn):
        COMPLEX_BUILDERS[name] = fn
        TEST_COMPLEX_SHAPES.append(name)
        return fn
    return decorator

# ==========================================
# 15 GMM Shapes (complex_gmm_1 to 15)
# ==========================================

@_register('complex_gmm_1')
def _build_gmm_1():
    # 3-arm spiral
    pts = []
    for arm in range(3):
        theta_offset = arm * (2 * np.pi / 3)
        for t in np.linspace(0.1, 1, 30):
            r = 0.4 * t
            theta = 3 * np.pi * t + theta_offset
            pts.append([0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)])
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)

@_register('complex_gmm_2')
def _build_gmm_2():
    # Concentric rings
    pts = []
    for t in np.linspace(0, 2*np.pi, 50):
        pts.append([0.5 + 0.3 * np.cos(t), 0.5 + 0.3 * np.sin(t)])
    for t in np.linspace(0, 2*np.pi, 30):
        pts.append([0.5 + 0.15 * np.cos(t), 0.5 + 0.15 * np.sin(t)])
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)

@_register('complex_gmm_3')
def _build_gmm_3():
    # Sine wave
    pts = []
    for x in np.linspace(0.1, 0.9, 60):
        y = 0.5 + 0.3 * np.sin(x * 4 * np.pi)
        pts.append([x, y])
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)

@_register('complex_gmm_4')
def _build_gmm_4():
    # Lemniscate (Figure 8)
    pts = []
    for t in np.linspace(0, 2*np.pi, 80):
        scale = 0.4 / (1 + np.sin(t)**2)
        x = 0.5 + scale * np.cos(t)
        y = 0.5 + scale * np.cos(t) * np.sin(t)
        pts.append([x, y])
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)

@_register('complex_gmm_5')
def _build_gmm_5():
    # Crescent moon
    pts = []
    for t in np.linspace(-np.pi/2, np.pi/2, 40):
        pts.append([0.6 + 0.3 * np.cos(t), 0.5 + 0.4 * np.sin(t)])
    for t in np.linspace(-np.pi/2, np.pi/2, 40):
        pts.append([0.45 + 0.15 * np.cos(t), 0.5 + 0.4 * np.sin(t)])
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)

@_register('complex_gmm_6')
def _build_gmm_6():
    # 4x4 dot grid
    pts = []
    for x in np.linspace(0.2, 0.8, 4):
        for y in np.linspace(0.2, 0.8, 4):
            pts.append([x, y])
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.03)

@_register('complex_gmm_7')
def _build_gmm_7():
    # Random scatter (stress test)
    np.random.seed(42)
    pts = np.random.uniform(0.1, 0.9, (20, 2))
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.02)

@_register('complex_gmm_8')
def _build_gmm_8():
    # 5-pointed star
    pts = []
    r_outer = 0.4
    r_inner = 0.15
    for i in range(11):
        angle = i * np.pi / 5 - np.pi/2
        r = r_outer if i % 2 == 0 else r_inner
        pts.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    dense_pts = []
    for i in range(10):
        p1 = np.array(pts[i])
        p2 = np.array(pts[i+1])
        for t in np.linspace(0, 1, 10):
            dense_pts.append(p1 + t*(p2-p1))
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(dense_pts, sigma=0.012)

@_register('complex_gmm_9')
def _build_gmm_9():
    # Parallel lines
    pts = []
    for y in [0.3, 0.5, 0.7]:
        for x in np.linspace(0.15, 0.85, 30):
            pts.append([x, y])
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)

@_register('complex_gmm_10')
def _build_gmm_10():
    # Hexagon
    pts = []
    hex_pts = []
    for i in range(7):
        angle = i * np.pi / 3
        hex_pts.append([0.5 + 0.35 * np.cos(angle), 0.5 + 0.35 * np.sin(angle)])
    for i in range(6):
        p1 = np.array(hex_pts[i])
        p2 = np.array(hex_pts[i+1])
        for t in np.linspace(0, 1, 12):
            pts.append(p1 + t*(p2-p1))
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)

@_register('complex_gmm_11')
def _build_gmm_11():
    # U-shape
    pts = []
    for y in np.linspace(0.8, 0.4, 20):
        pts.append([0.2, y])
    for t in np.linspace(np.pi, 2*np.pi, 30):
        pts.append([0.5 + 0.3 * np.cos(t), 0.4 + 0.3 * np.sin(t)])
    for y in np.linspace(0.4, 0.8, 20):
        pts.append([0.8, y])
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)

@_register('complex_gmm_12')
def _build_gmm_12():
    # Diamond
    pts = []
    diamond_pts = [[0.5, 0.1], [0.9, 0.5], [0.5, 0.9], [0.1, 0.5], [0.5, 0.1]]
    for i in range(4):
        p1 = np.array(diamond_pts[i])
        p2 = np.array(diamond_pts[i+1])
        for t in np.linspace(0, 1, 15):
            pts.append(p1 + t*(p2-p1))
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)

@_register('complex_gmm_13')
def _build_gmm_13():
    # Arrow
    pts = []
    for x in np.linspace(0.2, 0.7, 30):
        pts.append([x, 0.5])
    for t in np.linspace(0, 1, 15):
        pts.append([0.7 + t*(0.5-0.7), 0.5 + t*(0.7-0.5)])
    for t in np.linspace(0, 1, 15):
        pts.append([0.7 + t*(0.5-0.7), 0.5 + t*(0.3-0.5)])
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)

@_register('complex_gmm_14')
def _build_gmm_14():
    # Cross (X)
    pts = []
    for t in np.linspace(0.2, 0.8, 40):
        pts.append([t, t])
        pts.append([t, 1.0 - t])
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)

@_register('complex_gmm_15')
def _build_gmm_15():
    # Hourglass
    pts = []
    pts_corners = [[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8], [0.2, 0.2]]
    for i in range(4):
        p1 = np.array(pts_corners[i])
        p2 = np.array(pts_corners[i+1])
        for t in np.linspace(0, 1, 20):
            pts.append(p1 + t*(p2-p1))
    from shape_rasterizer import points_to_gmm
    return points_to_gmm(pts, sigma=0.015)


# ==========================================
# 15 Analytical Shapes (complex_ana_1 to 15)
# ==========================================

@_register('complex_ana_1')
def _build_ana_1(): # M
    return {'type': 'analytical', 'sigma': 0.025, 'segments': [
        ([0.2, 0.2], [0.2, 0.8]), ([0.2, 0.8], [0.5, 0.5]),
        ([0.5, 0.5], [0.8, 0.8]), ([0.8, 0.8], [0.8, 0.2])
    ]}

@_register('complex_ana_2')
def _build_ana_2(): # K
    return {'type': 'analytical', 'sigma': 0.025, 'segments': [
        ([0.3, 0.15], [0.3, 0.85]), ([0.7, 0.85], [0.3, 0.5]),
        ([0.3, 0.5], [0.7, 0.15])
    ]}

@_register('complex_ana_3')
def _build_ana_3(): # Z
    return {'type': 'analytical', 'sigma': 0.025, 'segments': [
        ([0.2, 0.8], [0.8, 0.8]), ([0.8, 0.8], [0.2, 0.2]),
        ([0.2, 0.2], [0.8, 0.2])
    ]}

@_register('complex_ana_4')
def _build_ana_4(): # E
    return {'type': 'analytical', 'sigma': 0.025, 'segments': [
        ([0.2, 0.15], [0.2, 0.85]), ([0.2, 0.85], [0.7, 0.85]),
        ([0.2, 0.5], [0.6, 0.5]), ([0.2, 0.15], [0.7, 0.15])
    ]}

@_register('complex_ana_5')
def _build_ana_5(): # Hexagram
    return {'type': 'analytical', 'sigma': 0.02, 'segments': [
        ([0.5, 0.9], [0.15, 0.3]), ([0.15, 0.3], [0.85, 0.3]), ([0.85, 0.3], [0.5, 0.9]),
        ([0.5, 0.1], [0.15, 0.7]), ([0.15, 0.7], [0.85, 0.7]), ([0.85, 0.7], [0.5, 0.1])
    ]}

@_register('complex_ana_6')
def _build_ana_6(): # Zigzag
    return {'type': 'analytical', 'sigma': 0.02, 'segments': [
        ([0.1, 0.2], [0.3, 0.8]), ([0.3, 0.8], [0.5, 0.2]),
        ([0.5, 0.2], [0.7, 0.8]), ([0.7, 0.8], [0.9, 0.2])
    ]}

@_register('complex_ana_7')
def _build_ana_7(): # Square Spiral Maze
    return {'type': 'analytical', 'sigma': 0.015, 'segments': [
        ([0.1, 0.1], [0.9, 0.1]), ([0.9, 0.1], [0.9, 0.9]),
        ([0.9, 0.9], [0.2, 0.9]), ([0.2, 0.9], [0.2, 0.25]),
        ([0.2, 0.25], [0.75, 0.25]), ([0.75, 0.25], [0.75, 0.75]),
        ([0.75, 0.75], [0.35, 0.75]), ([0.35, 0.75], [0.35, 0.4])
    ]}

@_register('complex_ana_8')
def _build_ana_8(): # TicTacToe
    return {'type': 'analytical', 'sigma': 0.02, 'segments': [
        ([0.33, 0.1], [0.33, 0.9]), ([0.66, 0.1], [0.66, 0.9]),
        ([0.1, 0.33], [0.9, 0.33]), ([0.1, 0.66], [0.9, 0.66])
    ]}

@_register('complex_ana_9')
def _build_ana_9(): # Bowtie
    return {'type': 'analytical', 'sigma': 0.02, 'segments': [
        ([0.1, 0.2], [0.1, 0.8]), ([0.1, 0.8], [0.5, 0.5]), ([0.5, 0.5], [0.1, 0.2]),
        ([0.9, 0.2], [0.9, 0.8]), ([0.9, 0.8], [0.5, 0.5]), ([0.5, 0.5], [0.9, 0.2])
    ]}

@_register('complex_ana_10')
def _build_ana_10(): # Cube projection
    return {'type': 'analytical', 'sigma': 0.015, 'segments': [
        ([0.2, 0.2], [0.6, 0.2]), ([0.6, 0.2], [0.6, 0.6]), ([0.6, 0.6], [0.2, 0.6]), ([0.2, 0.6], [0.2, 0.2]),
        ([0.4, 0.4], [0.8, 0.4]), ([0.8, 0.4], [0.8, 0.8]), ([0.8, 0.8], [0.4, 0.8]), ([0.4, 0.8], [0.4, 0.4]),
        ([0.2, 0.2], [0.4, 0.4]), ([0.6, 0.2], [0.8, 0.4]), ([0.6, 0.6], [0.8, 0.8]), ([0.2, 0.6], [0.4, 0.8])
    ]}

@_register('complex_ana_11')
def _build_ana_11(): # Crown
    return {'type': 'analytical', 'sigma': 0.02, 'segments': [
        ([0.1, 0.2], [0.9, 0.2]), ([0.1, 0.2], [0.1, 0.7]), ([0.9, 0.2], [0.9, 0.7]),
        ([0.1, 0.7], [0.3, 0.4]), ([0.3, 0.4], [0.5, 0.8]), ([0.5, 0.8], [0.7, 0.4]), ([0.7, 0.4], [0.9, 0.7])
    ]}

@_register('complex_ana_12')
def _build_ana_12(): # House
    return {'type': 'analytical', 'sigma': 0.02, 'segments': [
        ([0.2, 0.1], [0.8, 0.1]), ([0.8, 0.1], [0.8, 0.6]), ([0.8, 0.6], [0.2, 0.6]), ([0.2, 0.6], [0.2, 0.1]),
        ([0.2, 0.6], [0.5, 0.9]), ([0.5, 0.9], [0.8, 0.6]),
        ([0.4, 0.1], [0.4, 0.35]), ([0.6, 0.1], [0.6, 0.35]), ([0.4, 0.35], [0.6, 0.35])
    ]}

@_register('complex_ana_13')
def _build_ana_13(): # Asterisk
    return {'type': 'analytical', 'sigma': 0.02, 'segments': [
        ([0.1, 0.5], [0.9, 0.5]), ([0.5, 0.1], [0.5, 0.9]),
        ([0.2, 0.2], [0.8, 0.8]), ([0.2, 0.8], [0.8, 0.2])
    ]}

@_register('complex_ana_14')
def _build_ana_14(): # Envelope
    return {'type': 'analytical', 'sigma': 0.02, 'segments': [
        ([0.1, 0.2], [0.9, 0.2]), ([0.9, 0.2], [0.9, 0.8]), ([0.9, 0.8], [0.1, 0.8]), ([0.1, 0.8], [0.1, 0.2]),
        ([0.1, 0.8], [0.5, 0.5]), ([0.9, 0.8], [0.5, 0.5]),
        ([0.1, 0.2], [0.9, 0.8]), ([0.1, 0.8], [0.9, 0.2])
    ]}

@_register('complex_ana_15')
def _build_ana_15(): # Triforce
    return {'type': 'analytical', 'sigma': 0.02, 'segments': [
        ([0.1, 0.1], [0.9, 0.1]), ([0.9, 0.1], [0.5, 0.9]), ([0.5, 0.9], [0.1, 0.1]),
        ([0.3, 0.5], [0.7, 0.5]), ([0.7, 0.5], [0.5, 0.1]), ([0.5, 0.1], [0.3, 0.5])
    ]}
