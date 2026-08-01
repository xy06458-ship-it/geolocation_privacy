SENSITIVITY = [
    ('road',          0.10),  # 0
    ('sidewalk',      0.20),  # 1
    ('building',      0.90),  # 2
    ('wall',          0.70),  # 3
    ('fence',         0.55),  # 4
    ('pole',          0.50),  # 5
    ('traffic light', 0.80),  # 6
    ('traffic sign',  0.95),  # 7
    ('vegetation',    0.35),  # 8
    ('terrain',       0.25),  # 9
    ('sky',           0.05),  # 10
    ('person',        0.15),  # 11
    ('rider',         0.15),  # 12
    ('car',           0.10),  # 13
    ('truck',         0.10),  # 14
    ('bus',           0.12),  # 15
    ('train',         0.15),  # 16
    ('motorcycle',    0.10),  # 17
    ('bicycle',       0.10),  # 18
]
WEIGHTS = [s[1] for s in SENSITIVITY]
CLASS_NAMES = [s[0] for s in SENSITIVITY]