import matplotlib.pyplot as plt
from function.linear_least_squares import LinearLeastSquares
from data_helper import DATA_FILES
from algo_helper import run_sgd

DATA_FILE = DATA_FILES[0]
LR = 0.001
BATCH_SIZE = 100
LOSS = LinearLeastSquares()

_, accuracy_train, _ = run_sgd(DATA_FILE, LOSS, metric_fn=LOSS.loss, lr=LR, batch_size=BATCH_SIZE)
# show loss
fig, axs = plt.subplots(1, 1)
fig.suptitle(f'{LOSS.name}')
fig.tight_layout(pad=3.0)
axs.plot(accuracy_train, label='train loss')
axs.legend()
axs.set_title(f"{DATA_FILE}\n{LR=}\n{BATCH_SIZE=}")
axs.set_xlabel('epoch')
plt.show()
