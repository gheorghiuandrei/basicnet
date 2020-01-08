import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt


def animate_separation(X, y, model, epochs, name="animation", size=(8, 8)):
    X0, X1 = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 100),
        np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 100),
    )
    meshgrid = np.stack((X0.reshape(-1), X1.reshape(-1)), 1)
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    def func(frame):
        model.train(X, y, 1)
        a = model.predict(meshgrid).reshape(X0.shape)
        ax.cla()
        ax.axis("off")
        c = ax.contourf(X0, X1, a, alpha=0.6, cmap=plt.cm.plasma)
        paths = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.plasma)

        return c, paths

    ani = animation.FuncAnimation(fig, func, range(epochs), interval=100)
    ani.save(f"{name}.gif", writer="pillow")
