import foolbox
import keras
import numpy as np
from keras.models import load_model
from keras.datasets import mnist
from keras import backend as K

K.set_learning_phase(0)

model = load_model("mnist_resnet_mse.h5")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = np.reshape(x_test, (10000, 1, 28, 28));

# This works fine:
model.predict(np.reshape(x_test[0], (1,1,28,28)))

# foolbox code:
fmodel = foolbox.models.KerasModel(model, bounds=(0,255))
attack = foolbox.attacks.FGSM(fmodel)
adv = attack(x_test[0], y_test[0])

def get_ind(a):
    res = 0
    for i in range(len(a)):
        if a[i] > a[res]:
            res = i
    return res

i = 0
adv = attack(x_test[i], y_test[i])
pred = model.predict(np.reshape(adv, (1,1,28,28)))
print(get_ind(pred[0]))
print(y_test[i])

# Gradient attack
attack1 = foolbox.attacks.LBFGSAttack(fmodel)
adv1 = attack1(x_test[0], y_test[0])
pred = model.predict(np.reshape(adv1, (1,1,28,28)))
print(get_ind(pred[0]))
print(y_test[i])

# Gradient attack
attack2 = foolbox.attacks.LBFGSAttack(fmodel)
adv2 = attack2(x_test[0], y_test[0])
pred = model.predict(np.reshape(adv2, (1,1,28,28)))
print(get_ind(pred[0]))
print(y_test[i])