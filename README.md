# BasicNet

BasicNet is a basic neural network framework implemented in Python and NumPy.
Its sole purpose is to help me better understand the inner workings of neural
networks. I am putting it here so that it may help others as well.

## Installation

Clone with [git](https://git-scm.com/) and use
[conda](https://docs.conda.io/en/latest/) to install BasicNet.

```bash
git clone https://github.com/gheorghiuandrei/basicnet.git
cd basicnet
conda env create -f environment.yml
conda activate basic
pip install .
```

## Usage

```python
from sklearn.datasets import make_circles
from basic.activations import ReLU, Sigmoid
from basic.animations import animate_separation
from basic.layers import FC
from basic.losses import BinaryCrossEntropy
from basic.models import Model
from basic.optimizers import SGD


X, y = make_circles(1000, noise=0.1, random_state=0, factor=0.3)
network = [FC(2, 8), ReLU(), FC(8, 4), ReLU(), FC(4, 1), Sigmoid()]
model = Model(network, BinaryCrossEntropy(), SGD(0.001), batch_size=4)
animate_separation(X, y, model, epochs=100, name="circles")
```

<img src="circles.gif" height="400" width="400">