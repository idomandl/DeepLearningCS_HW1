import matplotlib.pyplot as plt
from function.softmax import Softmax
from data_helper import DATA_FILES
from algo_helper import run_sgd_nn
from nn import NN

DATA_FILE = DATA_FILES[0]
LR = 0.01
BATCH_SIZE = 100
LOSS = Softmax()

nn = NN([128, 64, 10], activation='relu') # todo: does sgd holds nn or the opposite?
_, accuracy_train, _ = run_sgd_nn(nn, DATA_FILE, LOSS, metric_fn=LOSS.loss, lr=LR, batch_size=BATCH_SIZE)
# show loss
fig, axs = plt.subplots(1, 1)
fig.suptitle(f'{LOSS.name}')
fig.tight_layout(pad=3.0)
axs.plot(accuracy_train, label='train loss')
axs.legend()
axs.set_title(f"{DATA_FILE}\n{LR=}\n{BATCH_SIZE=}")
axs.set_xlabel('epoch')
plt.show()
