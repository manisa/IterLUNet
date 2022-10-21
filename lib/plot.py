import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size' : 24})



def plot_loss_dice_history(history, experiment_name, graph_path):
    """
    Plots model training history 
    """

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(20,10))
    ax_loss.plot(history.epoch, history.history["loss"], label="train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="validation loss")
    ax_loss.legend()
    
    ax_acc.plot(history.epoch, history.history["dice_coef"], label="train dice_coef")
    ax_acc.plot(history.epoch, history.history["val_dice_coef"], label="validation dice_coef")
    ax_acc.legend()
    
    #fig.tight_layout()
    plt.suptitle('Dice loss and Dice_coeff graphs for ' + str(experiment_name), fontweight='regular', fontsize=30)
    plt.savefig(graph_path + '/' + str(experiment_name) + '_loss_dice' + '.png', format='png', dpi = 300)
    plt.close()
    return("DONE!")
    

def plot_dice_jacc_history(history, experiment_name, graph_path):
    """
    Plots model training history 
    """
    fig, (ax_dice, ax_jacc) = plt.subplots(1, 2, figsize=(20,10))
    ax_dice.plot(history.epoch, history.history["dice_coef"], label="train F1 Score")
    ax_dice.plot(history.epoch, history.history["val_dice_coef"], label="validation F1 Score")
    ax_dice.legend()
    
    ax_jacc.plot(history.epoch, history.history["jaccard"], label="train jaccard")
    ax_jacc.plot(history.epoch, history.history["val_jaccard"], label="validation jaccard")
    ax_jacc.legend()
    
    #fig.tight_layout()
    plt.suptitle('F1 score and IoU graphs for ' + str(experiment_name), fontweight='regular', fontsize=30)
    plt.savefig(graph_path + '/' + str(experiment_name) + '_f1_iou' + '.png', format='png', dpi = 300)
    plt.close()
    return("DONE!")