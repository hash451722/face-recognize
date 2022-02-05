import cv2
import numpy as np
import matplotlib.pyplot as plt


class Kmeans():
    def __init__(self):
        self.flags = cv2.KMEANS_RANDOM_CENTERS

    def apply(self, data, nclusters, iter=10, eps=0.1, attempts=10):
        data = np.float32(data)  # convert to np.float32
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iter, eps)
        compactness, labels, centers = cv2.kmeans(data,
                                                  nclusters,
                                                  None,
                                                  criteria,
                                                  attempts,
                                                  self.flags)

        return compactness, labels, centers


if __name__ == '__main__':
    x = np.random.randint(25, 50, (25,2))
    y = np.random.randint(60, 85, (25,2))
    data = np.vstack((x, y))
    print(data.shape)

    kmeans = Kmeans()
    compactness, labels, centers = kmeans.apply(data, 2)

    # Now separate the data, Note the flatten()
    A = data[labels.ravel()==0]
    B = data[labels.ravel()==1]

    # Plot the data
    plt.scatter(A[:,0], A[:,1])
    plt.scatter(B[:,0], B[:,1], c='r')
    plt.scatter(centers[:,0], centers[:,1], s=80, c='y', marker='s')
    plt.xlabel('Height'), plt.ylabel('Weight')
    plt.show()


# https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
