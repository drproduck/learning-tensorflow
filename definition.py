import numpy as np
import numpy.linalg as lin
def whitening(x):
    n,_ = x.shape
    SIG = (1.0 / np.sqrt(n)) * x
    eivec, eival, _ = lin.svd(SIG)
    whitener = (np.diag(eival ** (-1)), eivec.T)
    return whitener

def main():
    x = np.array([[1,2,3,4],[5,6,7,8],[9,10,13,14],[20,19,18,87]], dtype=np.float64).T
    scale, project = whitening(x)
    print(scale)
    print(project)
    print(project.dot(x))
    y = project.dot(x)
    print(0.25* y.dot(y.T))
    z = scale.dot(project.dot(x))
    print(0.25 * z.dot(z.T))

if __name__ == '__main__':
    main()
