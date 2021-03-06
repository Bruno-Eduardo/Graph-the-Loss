import matplotlib.pyplot as plt
import numpy as np
import re


########## Process data
def load_csv():
    train_loss = np.loadtxt('data.csv', delimiter=',')
    eval_loss = np.loadtxt('valLoss.csv', delimiter=',')
    return train_loss, eval_loss


def parse_data_from_txt(filename):
    loss = []
    acc = []
    val_loss = []
    val_acc = []

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if "loss" not in line:
            continue

        splitted_line = re.split(' - ', line.replace('\n', ''))

        for sliced in splitted_line:
            if sliced.startswith('loss'):
                loss.append(float(re.findall("\d+\.\d+", sliced)[0]))
            if sliced.startswith('acc'):
                acc.append(float(re.findall("\d+\.\d+", sliced)[0]))
            if sliced.startswith('val_loss'):
                val_loss.append(float(re.findall("\d+\.\d+", sliced)[0]))
            if sliced.startswith('val_categorical_accuracy'):
                val_acc.append(float(re.findall("\d+\.\d+", sliced)[0]))

    return loss, acc, val_loss, val_acc


########## Handle plot
def close_event():
    plt.close()  # timer calls this function after 3 seconds and closes the window


def plot_set_timeout(fig, time_ms, callback=close_event):
    timer = fig.canvas.new_timer(interval=time_ms)
    timer.add_callback(callback)
    return timer


def get_horizontal_axis(train_loss, eval_loss):
    x_axis_train = np.arange(1, len(train_loss) + 1)
    x_axis_eval = np.arange(1, len(eval_loss) + 1) * len(train_loss) / (len(eval_loss))
    return x_axis_train, x_axis_eval


def plot_loss(train_loss, eval_loss):
    x_axis_train, x_axis_eval = get_horizontal_axis(train_loss, eval_loss)

    fig = plt.figure()
    t = plot_set_timeout(fig, 1000, callback=close_event)

    plt.title('Loss through epochs')
    plt.plot(x_axis_train, train_loss)
    plt.plot(x_axis_eval, eval_loss)
    plt.ylim([0, max([max(train_loss), max(eval_loss)])])

    t.start()  # start timer of close callback
    plt.show()


def loop_parse_and_plot(filename, n_iter=-1):
    def loop():
        l, a, vl, va = parse_data_from_txt(filename)
        plot_loss(l, vl)

    if n_iter < 1:
        while True:
            loop()
    else:
        for _ in range(n_iter):
            loop()


########################
if __name__ == '__main__':
    loop_parse_and_plot("C:\\Users\\Bruno Eduardo\\Desktop\\graphLoss\\dataFromPyCharm.txt", n_iter=-1)
