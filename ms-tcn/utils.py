import re
import matplotlib.pyplot as plt
import pandas as pd


def get_data(file_path):
    epoch_pattern = "[epoch (\d+)]:"
    train_pattern = "train_loss = (.*?), train_acc"
    val_pattern = "validation_loss = (.*?), validation_acc ="
    valacc_pattern = ", validation_acc = (\d+\.\d+)"
    trainacc_pattern = ", train_acc = (\d+\.\d+), validation_loss"
    
    data = pd.DataFrame()

    lines=[]
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            lines.append(line.strip())
            line = fp.readline()
    data['lines'] = lines

    def get_Value(pattern, s):
        value = re.search(pattern, s)
        if value:
            return float(value.group(1))
        else: return None

    data['epoch'] = data['lines'].apply(lambda x: get_Value(epoch_pattern, x))
    data['train_loss'] = data['lines'].apply(lambda x: get_Value(train_pattern, x))
    data['val_loss'] = data['lines'].apply(lambda x: get_Value(val_pattern, x))
    data['val_acc'] = data['lines'].apply(lambda x: get_Value(valacc_pattern, x))
    data['train_acc'] = data['lines'].apply(lambda x: get_Value(trainacc_pattern, x))
    
    return data



def visualize_train(data):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('itr')
    ax1.set_ylabel('training loss', color=color)
    ax1.plot(data['train_loss'].dropna(), color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 

    color = 'tab:blue'
    ax2.set_ylabel('training accuracy', color=color)  
    ax2.plot(data['train_acc'].dropna(), color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.savefig("train_acc_vs_loss.png", dpi=300)
    
    
    
def visualize_val(data):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('itr')
    ax1.set_ylabel('validation loss', color=color)
    ax1.plot(data['val_loss'].dropna(), color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 

    color = 'tab:blue'
    ax2.set_ylabel('validation accuracy', color=color)  
    ax2.plot(data['val_acc'].dropna(), color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.savefig("val_acc_vs_loss.png", dpi=300)
    

    
def visualize(record_file):
    data = get_data(record_file + '.txt')
    visualize_train(data)
    visualize_val(data)
    

