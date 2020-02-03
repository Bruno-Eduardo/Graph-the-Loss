import matplotlib.pyplot as plt
import numpy as np


def load_csv():
    train_loss = np.loadtxt('data.csv', delimiter=',')
    eval_loss = np.loadtxt('valLoss.csv', delimiter=',')
    return train_loss, eval_loss


def get_horizontal_axis(train_loss, eval_loss):
    x_axis_train = np.arange(1, len(train_loss) + 1)
    x_axis_eval = np.arange(1, len(eval_loss) + 1) * len(train_loss) / (len(eval_loss))
    return x_axis_train, x_axis_eval


def plot_loss(train_loss, eval_loss):
    x_axis_train, x_axis_eval = get_horizontal_axis(train_loss, eval_loss)

    plt.figure()
    plt.title('Loss through epochs')
    plt.plot(x_axis_train, train_loss)
    plt.plot(x_axis_eval, eval_loss)
    plt.ylim([0, max([max(train_loss), max(eval_loss)])])
    plt.show()


if __name__ == '__main__':
    train_loss_csv, eval_loss_csv = load_csv()
    plot_loss(train_loss_csv, eval_loss_csv)
