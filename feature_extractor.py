import numpy as np

def extract_features(image, dem, p, q):
    # ensure all shapes match
    assert image.shape == dem.shape == p.shape == q.shape

    slope = np.sqrt(p**2 + q**2)

    return np.column_stack([
        image.reshape(-1),
        dem.reshape(-1),
        p.reshape(-1),
        q.reshape(-1),
        slope.reshape(-1)
    ])
