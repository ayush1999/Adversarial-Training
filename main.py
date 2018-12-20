import numpy as np
np.random.seed(2142)
from keras.models import Input
from keras.losses import mean_squared_error as scc
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K

#load pretrained model
model = load_model('mnist_resnet_mse_normalized.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = np.reshape(x_test, (10000,1,28,28))/255
y_test = to_categorical(y_test)

def get_ind(a):
    res = 0
    for i in range(len(a)):
        if a[i] > a[res]:
            res = i
    return res

# Returns the adversarial image from x_test[i]
def adversify(i):
    c = x_test[i]
    y = Input(shape=y_test[i].shape)
    grads = K.gradients(scc(y, model.output), model.inputs)
    func = K.function(model.inputs + [y, K.learning_phase()], grads)
    
    for i in range(50):
        out = func([np.reshape(c, (1,1,28,28)), np.reshape(y_test[i], (1,10)), 0])
        c -= 0.2*out[0][0]
        
    return c

img = adversify(0)

if get_ind(model.predict(np.reshape(img, (1,1,28,28)))[0] )== get_ind(y_test[0]):
    print("still gives correct output")
else:
    print("adversarial example works")
