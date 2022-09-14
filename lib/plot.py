import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def plot_loss_dice_history(history, experiment_name):
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
    plt.suptitle('dice loss and dice_coeff graphs for ' + str(experiment_name), fontweight='regular', fontsize=24)
    plt.savefig(str(experiment_name) + '_loss_dice' + '.png', format='png')
    plt.close()
    

def plot_dice_jacc_history(history, experiment_name):
    """
    Plots model training history 
    """
    fig, (ax_dice, ax_jacc) = plt.subplots(1, 2, figsize=(20,10))
    ax_dice.plot(history.epoch, history.history["dice_coef"], label="train dice_coef")
    ax_dice.plot(history.epoch, history.history["val_dice_coef"], label="validation dice_coef")
    ax_dice.legend()
    
    ax_jacc.plot(history.epoch, history.history["jaccard"], label="train jaccard")
    ax_jacc.plot(history.epoch, history.history["val_jaccard"], label="validation jaccard")
    ax_jacc.legend()
    
    #fig.tight_layout()
    plt.suptitle('iou and F1 score graphs for ' + str(experiment_name), fontweight='regular', fontsize=24)
    plt.savefig(str(experiment_name) + '_iou_f1' + '.png', format='png')
    plt.close()