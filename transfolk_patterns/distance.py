import math


def distanceFunction(point1, point2, l):
    buf = 0.0
    for i in range(l - 1):
        buf = buf + pow(point1[i] - point2[i], 2)
    return pow(buf, 0.5)


def sortingFunction(N, M, i, j):
    Sigma = N / 6.0
    A = 1.0
    return A * math.exp(-0.5 * pow(i - (N - 1) * j / (M - 1), 2) / pow(Sigma, 2))

def CalculateDistance(points, centroids):
    n = len(points)
    c = len(centroids)
    d = len(points)

    distance = 0
    if n==c:
        for i in range(n - 1):
            distance = distance + distanceFunction(points[i], centroids[i], len(centroids[i]))
        return distance / c / n
    else:
        for i in range(n - 1):
            for j in range(c - 1):
                distance = distance +  sortingFunction(n, c, i, j) * distanceFunction(points[i], centroids[j], len(centroids[j]))
        return distance / c / n


