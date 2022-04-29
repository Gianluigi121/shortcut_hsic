from model import PretrainedDenseNet121
from data_builder import load_ds_from_csv
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import argparse
import matplotlib.pyplot as plt

# Define our metrics
# MAIN_DIR = '/nfs/turbo/coe-rbg/zhengji/age_shortcut/'
# train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
# train_accuracy = tf.keras.metrics.Accuracy('train_accuracy')
# # train_auroc = tf.keras.metrics.AUC('train_auroc')
# eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
# eval_accuracy = tf.keras.metrics.Accuracy('eval_accuracy')
# # eval_auroc = tf.keras.metrics.AUC('eval_auroc')
# # test_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
# # test_accuracy = tf.keras.metrics.Accuracy('eval_accuracy')
# # test_auroc = tf.keras.metrics.AUC('eval_auroc')

# # Set up summary writers to write the summaries to disk in a different logs directory
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# eval_log_dir = 'logs/gradient_tape/' + current_time + '/eval'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)


def auroc(auc_metric, labels, predictions):
    """ Computes AUROC """
    auc_metric(y_true=labels, y_pred=predictions)
    return auc_metric.result()

def accuracy(acc, labels, predictions):
    """ Computes Accuracy"""
    acc(y_true=labels, y_pred=predictions)
    return acc.result()

def update_eval_metrics_dict(acc, auc, labels, predictions):
    y_main = tf.expand_dims(labels[:, 0], axis=-1)

    eval_metrics_dict = {}
    eval_metrics_dict['accuracy'] = accuracy(acc,
        labels=y_main, predictions=predictions["classes"])
    eval_metrics_dict["auc"] = auroc(auc, 
        labels=y_main, predictions=predictions["probabilities"])

    return eval_metrics_dict

def hsic(x, y, sample_weights, sigma=1.0):
	""" Computes the weighted HSIC between two arbitrary variables x, y"""

	if len(x.shape) == 1:
		x = tf.expand_dims(x, axis=-1)

	if len(y.shape) == 1:
		y = tf.expand_dims(y, axis=-1)

	if sample_weights == None:
		sample_weights = tf.ones((tf.shape(y)[0], 1))

	if len(sample_weights.shape) == 1:
		sample_weights = tf.expand_dims(sample_weights, axis=-1)

	sample_weights_T = tf.transpose(sample_weights)

	kernel_fxx = tfp.math.psd_kernels.ExponentiatedQuadratic(
			amplitude=1.0, length_scale=sigma)

	kernel_xx = kernel_fxx.matrix(x, x)
	kernel_fyy = tfp.math.psd_kernels.ExponentiatedQuadratic(
			amplitude=1.0, length_scale=sigma)
	kernel_yy = kernel_fyy.matrix(y, y)

	N = tf.cast(tf.shape(y)[0], tf.float32)

	# First term
	hsic_1 = tf.math.multiply(kernel_xx, kernel_yy)
	hsic_1 = tf.linalg.matmul(sample_weights_T, hsic_1)
	hsic_1 = tf.linalg.matmul(hsic_1, sample_weights)
	hsic_1 = hsic_1 / (N **2)

	# Second term
	# Note there is a typo in the paper. Authors will update
	W_matrix = tf.linalg.matmul(sample_weights, sample_weights_T)
	hsic_2 = tf.math.multiply(kernel_yy, W_matrix)
	hsic_2 = tf.reduce_sum(hsic_2, keepdims=True) * tf.reduce_sum(kernel_xx, keepdims=True)
	hsic_2 = hsic_2 / (N ** 4)

	# third term
	hsic_3 = tf.linalg.matmul(kernel_yy, sample_weights)
	hsic_3 = tf.math.multiply(
			tf.reduce_sum(kernel_xx, axis =1, keepdims=True), hsic_3)
	hsic_3 = tf.linalg.matmul(sample_weights_T, hsic_3)
	hsic_3 = 2 * hsic_3 / (N ** 3)

	hsic_val = hsic_1 + hsic_2 - hsic_3
	hsic_val = tf.maximum(0.0, hsic_val)
	return hsic_val

def compute_loss(labels, logits, embedding, params):
    prediction_loss, hsic_loss = compute_loss_unweighted(labels, logits,
        embedding, params)
    return prediction_loss, hsic_loss

def compute_loss_unweighted(labels, logits, embedding, params):
    # labels: ground truth labels([y0(pnemounia), age])
    # logits: predicted label(pnemounia)
    # embedding: a learned representation vector
    y_main = tf.expand_dims(labels[:, 0], axis=-1)

    individual_losses = tf.keras.losses.binary_crossentropy(y_main, logits,
        from_logits=True)

    unweighted_loss = tf.reduce_mean(individual_losses)
    aux_y = labels[:, 1:]
    if params['alpha'] > 0:
        hsic_loss = hsic(embedding, aux_y, sample_weights=None,
            sigma=params['sigma'])
    else:
        hsic_loss = 0.0
    return unweighted_loss, hsic_loss


# Training function for one step
def train_step(model, optimizer, features, labels, params):
    with tf.GradientTape() as tape:
        logits, zpred = model(features)
        # y_pred: predicted probability
        ypred = tf.nn.sigmoid(logits)

        predictions = {
            "classes": tf.cast(tf.math.greater_equal(ypred, .5), dtype=tf.float32),
            "logits": logits,
            "probabilities": ypred,
            "embedding": zpred
        }

        prediction_loss, hsic_loss = compute_loss(labels, logits, zpred, params)
        regularization_loss = tf.reduce_sum(model.losses)
        loss = regularization_loss + prediction_loss + params["alpha"] * hsic_loss
    
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # metrics: accuracy(class), auc(probability)
    train_loss(loss)
    metrics = update_eval_metrics_dict(train_accuracy, train_auroc, labels, predictions)
    acc = metrics['accuracy']
    auroc = metrics["auc"]
    print(f"Training auroc: {auroc}")
    return loss, acc, auroc

def eval_step(model, x, y, params):
    main_eval_metrics = {}
    # logits: prediction, zpred: representation vec
    logits, zpred = model(x)
    # y_pred: predicted probability
    ypred = tf.nn.sigmoid(logits)
    predictions = {
        "classes": tf.cast(tf.math.greater_equal(ypred, .5), dtype=tf.float32),
        "logits": logits,
        "probabilities": ypred,
        "embedding": zpred
    }

    # loss
    eval_pred_loss, eval_hsic_loss = compute_loss_unweighted(y, logits, zpred, params)
    loss = (eval_pred_loss + params["alpha"] * eval_hsic_loss).numpy()
    eval_loss(loss)
    # main_eval_metrics['pred_loss'] = tf.compat.v1.metrics.mean(eval_pred_loss)
    # main_eval_metrics['hsic'] = tf.compat.v1.metrics.mean(eval_hsic)
    
    # metrics
    metrics = update_eval_metrics_dict(eval_accuracy, eval_auroc, y, predictions)
    acc = metrics['accuracy']
    auroc = metrics["auc"]
    print(f"Valid auroc: {auroc}")
    return loss, acc, auroc

def train_eval(params, train_ds, valid_ds):
    # Define dataset, model and optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
    model = PretrainedDenseNet121(embedding_dim=params['embedding_dim'], l2_penalty=params['l2_penalty'])
    # train_ds, valid_ds,  = load_ds_from_csv(MAIN_DIR+'data', 'True', params)

    train_loss_list = []
    eval_loss_list = []
    train_acc_list = []
    eval_acc_list = []
    train_auroc_list = []
    eval_auroc_list = []
    for epoch in range(params['epoch_num']):
        print(f"Epoch Number: {epoch}")
        print("Training")
        train_sample_sum = 0
        train_loss_sum = 0
        train_acc_sum = 0
        train_auroc_sum = 0
        for step, (x, y) in enumerate(train_ds):
            num_sample = x.shape[0]
            train_step_loss, train_step_acc, train_step_auroc = train_step(model, optimizer, x, y, params)
            train_sample_sum += num_sample
            train_loss_sum += train_step_loss*num_sample
            train_acc_sum += train_step_acc*num_sample
            train_auroc_sum += train_step_auroc*num_sample

        train_avg_loss = train_loss_sum / train_sample_sum
        train_avg_acc = train_acc_sum / train_sample_sum
        train_avg_auroc = train_auroc_sum / train_sample_sum
        
        print("Evaluating")
        eval_sample_sum = 0
        eval_loss_sum = 0
        eval_acc_sum = 0
        eval_auroc_sum = 0
        for step, (x, y) in enumerate(valid_ds):
            num_sample = x.shape[0]
            eval_step_loss, eval_step_acc, eval_step_auroc = eval_step(model, x, y, params)
            eval_sample_sum += num_sample
            eval_loss_sum += eval_step_loss*num_sample
            eval_acc_sum += eval_step_acc*num_sample
            eval_auroc_sum += eval_step_auroc*num_sample
        
        eval_avg_loss = eval_loss_sum / eval_sample_sum
        eval_avg_acc = eval_acc_sum / eval_sample_sum
        eval_avg_auroc = eval_auroc_sum / eval_sample_sum
        
        train_loss_list.append(float(train_avg_loss))
        train_acc_list.append(float(train_avg_acc))
        train_auroc_list.append(float(train_avg_auroc))
        eval_loss_list.append(float(eval_avg_loss))
        eval_acc_list.append(float(eval_avg_acc))
        eval_auroc_list.append(float(eval_avg_auroc))

        # # Reset metrics every epoch
        train_loss.reset_states()
        eval_loss.reset_states()
        train_accuracy.reset_states()
        eval_accuracy.reset_states()
        train_auroc.reset_states()
        eval_auroc.reset_states()
    
    result_dict = {"train_loss": train_loss_list,
                   "train_acc": train_acc_list,
                   "train_auroc": train_auroc_list,
                   "eval_loss": eval_loss_list,
                   "eval_acc": eval_acc_list,
                   "eval_auroc": eval_auroc_list
                  }
    
    return model, result_dict

def test(params, test_ds, model):
    print("Testing")
    test_sample_sum = 0
    test_loss_sum = 0
    test_acc_sum = 0
    test_auroc_sum = 0
    for step, (x, y) in enumerate(test_ds):
        num_sample = x.shape[0]
        test_step_loss, test_step_acc, test_step_auroc = eval_step(model, x, y, params)
        test_sample_sum += num_sample
        test_loss_sum += test_step_loss*num_sample
        test_acc_sum += test_step_acc*num_sample
        test_auroc_sum += test_step_auroc*num_sample

    test_avg_loss = test_loss_sum / test_sample_sum
    test_avg_acc = test_acc_sum / test_sample_sum
    test_avg_auroc = test_auroc_sum / test_sample_sum

    return test_avg_loss, test_avg_acc, test_avg_auroc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--pixel", type=int, default=128, help="Image size: pixel*pixel")
    parser.add_argument("--embedding_dim", type=int, default=-1, help="Embedding dim for the feature vector")
    parser.add_argument("--l2_penalty", type=float, default=0.01, help="Regularizer on each layer")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--alpha", type=int, default=100, help="Alpha for hsic loss")
    parser.add_argument("--sigma", type=float, default=0.01, help="Sigma used in hsic")
    args = parser.parse_args()

    MAIN_DIR = "/nfs/turbo/coe-rbg/zhengji/age_shortcut/"
    params = {'epoch_num': args.epoch_num,
            'lr': args.lr,
            'l2_penalty': args.l2_penalty,
            'alpha': args.alpha,
            'sigma': args.sigma,
            'batch_size': args.batch_size,
            'pixel': args.pixel,
            'embedding_dim': args.embedding_dim,
            }
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.Accuracy('train_accuracy')
    train_auroc = tf.keras.metrics.AUC()
    eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
    eval_accuracy = tf.keras.metrics.Accuracy('eval_accuracy')
    eval_auroc = tf.keras.metrics.AUC()

    # Construct the dataset
    train_ds, valid_ds, unskew_test_ds, skew_test_ds = load_ds_from_csv(MAIN_DIR+'data', 'True', params)
    model, result_dict = train_eval(params, train_ds, valid_ds)
    print(result_dict)

    print()
    print("Test result")
    unskew_test_loss, unskew_test_acc, unskew_test_auroc = test(params, unskew_test_ds, model)
    print(f"Unskew Test loss: {unskew_test_loss}, Unskew test acc: {unskew_test_acc}, Unskew test auroc: {unskew_test_auroc}")
    skew_test_loss, skew_test_acc, skew_test_auroc = test(params, skew_test_ds, model)
    print(f"Skew Test loss: {skew_test_loss}, Skew test acc: {skew_test_acc}, Skew test auroc: {skew_test_auroc}")

    # Plotting
    epoch_num = args.epoch_num
    l2 = args.l2_penalty
    epochs = range(len(result_dict["train_loss"]))

    # Plot 1: Training loss
    plt.figure()
    plt.plot(epochs, result_dict["train_loss"], '--o', label='Training loss')
    plt.xlabel("Training loss")
    plt.ylabel("Epoch")
    plt.title('Training loss VS Epoch')
    plt.savefig(MAIN_DIR + f'plot_unw/train_loss_epoch{epoch_num}_l2{l2}.png')

    # Plot 2: Training accuracy
    plt.figure()
    plt.plot(epochs, result_dict["train_acc"], '--o', label='Training Accuracy')
    plt.xlabel("Training Accuracy")
    plt.ylabel("Epoch")
    plt.title('Training Accuracu VS Epoch')
    plt.savefig(MAIN_DIR + f'plot_unw/train_acc_epoch{epoch_num}_l2{l2}.png')

    # Plot 3: Training AUROC
    plt.figure()
    plt.plot(epochs, result_dict["train_auroc"], '--o', label='Training Auroc')
    plt.xlabel("Training Auroc")
    plt.ylabel("Epoch")
    plt.title('Training Auroc VS Epoch')
    plt.savefig(MAIN_DIR + f'plot_unw/train_auroc_epoch{epoch_num}_l2{l2}.png')

    # Plot 4: Validation loss
    plt.figure()
    plt.plot(epochs, result_dict["eval_loss"], '--o', label='Validation loss')
    plt.xlabel("Validation loss")
    plt.ylabel("Epoch")
    plt.title('Validation loss VS Epoch')
    plt.savefig(MAIN_DIR + f'plot_unw/val_loss_epoch{epoch_num}_l2{l2}.png')

    # Plot 5: Validation Accuracy
    plt.figure()
    plt.plot(epochs, result_dict["eval_acc"], '--o', label='Validation Accuracy')
    plt.xlabel("Validation Accuracy")
    plt.ylabel("Epoch")
    plt.title('Validation Accuracy VS Epoch')
    plt.savefig(MAIN_DIR + f'plot_unw/val_acc_epoch{epoch_num}_l2{l2}.png')

    # Plot 6: Validation Auroc
    plt.figure()
    plt.plot(epochs, result_dict["eval_auroc"], '--o', label='Validation Auroc')
    plt.xlabel("Validation Auroc")
    plt.ylabel("Epoch")
    plt.title('Validation Auroc VS Epoch')
    plt.savefig(MAIN_DIR + f'plot_unw/val_auroc_epoch{epoch_num}_l2{l2}.png')