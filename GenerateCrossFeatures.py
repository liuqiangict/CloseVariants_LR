
import sys
import numpy as np

class GenerateCrossFeatures():
    def __init__(self):
        pass

    def GetFeature(self, a, b):
        if isinstance(a, str):
            ta = a.strip()
        else:
            ta = a.decode('utf-8').strip()
        if isinstance(b, str):
            tb = b.strip()
        else:
            tb = b.decode('utf-8').strip()
        features = []
        for i in range(min(len(ta), 63)):
            for j in range(min(len(tb), 63)):
                orda = min(ord(ta[i]), 255)
                ordb = min(ord(tb[j]), 255)
                features.append((i << 22)| (orda << 14) | (j << 8) | (ordb))
        return ";".join([str(features[i]) for i in range(len(features))])


if __name__ == '__main__':
    m = GenerateCrossFeatures()
    features = m.GetFeature("a", "a")
    print(features)
