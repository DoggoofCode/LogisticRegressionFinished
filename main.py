import numpy as np


def sigmoid(value: float) -> float:
    return 1 / (1 + np.exp(-value))


def fitting(**kwargs):
    global coefficients
    for _ in range(kwargs["EPOCHS"]):
        z = np.dot(kwargs["x"], coefficients)
        h = sigmoid(z)
        gradient = np.dot(kwargs['x'].T, (h - kwargs['y'])) / 3
        # gradient = [index_number for index_number in range(len(kwargs['y']))]
        coefficients -= gradient * kwargs["lr"]


def main(**kwargs):
    x_training = kwargs['x']
    y_training = kwargs['y']
    fitting(x=x_training, y=y_training, EPOCHS=kwargs['epochs'], lr=kwargs["lr"])


def predict(x: np.array, threshold=0.5):
    z = np.dot(x, coefficients)
    probability = sigmoid(z)
    return (probability >= threshold).astype(int)


if __name__ == "__main__":
    global coefficients
    X_train = np.array([
        [0.2, 0.3, 0.1],
        [0.1, 0.2, 0.4],
        [0.4, 0.5, 0.2],
        [0.5, 0.3, 0.1],
    ])

    Y_train = np.array([0, 1, 0, 1])

    LR = 0.01
    EPOCHS = 1000
    m, n = X_train.shape
    coefficients = np.zeros(n)

    main(x=X_train, y=Y_train, lr=LR, epochs=EPOCHS)
    print(predict(np.array([float(input("What is value 1: ")),
                            float(input("What is value 2: ")),
                            float(input("What is value 3: "))]
                           )
                  )
          )
