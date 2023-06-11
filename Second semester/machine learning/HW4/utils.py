from itertools import product
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, confusion_matrix, accuracy_score
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def calculate_scores(y_true, y_pred, minority_label):
    precision, recall, f_measure, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', \
                                                                      pos_label=minority_label, zero_division=True)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    g_mean = 0
    if(tp != 0 and (tn + fp) != 0 and tn != 0 and (tp + fn) != 0):
        g_mean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

    return precision, recall, f_measure, g_mean, accuracy

def calculate_auc(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = np.mean([roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(n_classes)])
    return auc

def plot_roc_curve(fpr, tpr, auc, keys, params):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = ""
    for i in range(len(keys)):
        title += str(keys[i]) + ": " + str(params[i]) + ", "
    plt.title(f'Receiver Operating Characteristic ({title})')
    plt.legend(loc='lower right')
    plt.show()

def perform_grid_search(classifier, param_grid, X_train, y_train, X_test, y_test, n_iterations=5, \
                        report_best=False, plot_roc=False, print_std=True):
    unique_labels, label_counts = np.unique(y_train, return_counts=True)
    minority_label = unique_labels[np.argmin(label_counts)]
    best_score = -1
    best_params = {}

    mean_precision = []
    mean_recall = []
    mean_f_measure = []
    mean_g_mean = []
    mean_auc = []
    mean_fpr = []
    mean_tpr = []
    mean_accuracy = []
    std_precision = []
    std_recall = []
    std_f_measure = []
    std_g_mean = []
    std_auc = []
    std_accuracy = []

    # Generate all possible combinations of hyperparameters
    param_combinations = list(product(*param_grid.values()))

    # Iterate over each combination
    for params in param_combinations:
        # Set the classifier's hyperparameters
        for param_name, param_value in zip(param_grid.keys(), params):
            setattr(classifier, param_name, param_value)

        all_precision = []
        all_recall = []
        all_f_measure = []
        all_g_mean = []
        all_auc = []
        all_fpr = []
        all_tpr = []
        all_accuracy = []

        # Perform multiple iterations
        for _ in range(n_iterations):
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)

            y_test_ = y_test
            y_pred_ = y_pred
            minority_label_ = minority_label
            if(len(np.unique(y_test)) > 2):
                y_test_ = np.select([y_test == minority_label, y_test != minority_label], [1, 0], y_test)
                y_pred_ = np.select([y_pred == minority_label, y_pred != minority_label], [1, 0], y_pred)
                unique_labels, label_counts = np.unique(y_pred_, return_counts=True)
                minority_label_ = unique_labels[np.argmin(label_counts)]

            precision, recall, f_measure, g_mean, accuracy = calculate_scores(y_test_, y_pred_, minority_label_)

            auc = calculate_auc(y_test_, y_pred_)

            fpr, tpr, _ = roc_curve(y_test_, y_pred_)

            # Store precision, recall, F-measure, G-mean, and AUC for each iteration
            all_precision.append(precision)
            all_recall.append(recall)
            all_f_measure.append(f_measure)
            all_g_mean.append(g_mean)
            all_accuracy.append(accuracy)
            all_auc.append(auc)
            all_fpr.append(fpr)
            all_tpr.append(tpr)

        # Calculate the mean of precision, recall, F-measure, G-mean, and AUC for the current configuration
        mean_precision.append(np.mean(all_precision))
        mean_recall.append(np.mean(all_recall))
        mean_f_measure.append(np.mean(all_f_measure))
        mean_g_mean.append(np.mean(all_g_mean))
        mean_accuracy.append(np.mean(all_accuracy))
        mean_auc.append(np.mean(all_auc))
        mean_fpr.append(np.mean(all_fpr, axis=0))
        mean_tpr.append(np.mean(all_tpr, axis=0))
        std_precision.append(np.std(all_precision))
        std_recall.append(np.std(all_recall))
        std_f_measure.append(np.std(all_f_measure))
        std_g_mean.append(np.std(all_g_mean))
        std_accuracy.append(np.std(all_accuracy))
        std_auc.append(np.std(all_auc))

        # Check if the current score is better than the best score
        if np.mean(all_g_mean) > best_score:
            best_score = np.mean(all_g_mean)
            best_params = {param_name: param_value for param_name, param_value in zip(param_grid.keys(), params)}

    mean_table = PrettyTable()
    mean_table.field_names = list(param_grid.keys()) + ["Precision (Mean)", "Recall (Mean)", "F-measure (Mean)", \
                                                    "G-mean (Mean)", "AUC (Mean)", "Accuracy (Mean)"]
    
    for i, params in enumerate(param_combinations):
        row = list(params) + [round(mean_precision[i], 4), round(mean_recall[i], 4), round(mean_f_measure[i], 4),\
                                            round(mean_g_mean[i], 4), round(mean_auc[i], 4), round(mean_accuracy[i], 4)]
        mean_table.add_row(row)

    print("Mean Evaluation Metrics:")
    print(mean_table)

    if(print_std):
        std_table = PrettyTable()
        std_table.field_names = list(param_grid.keys()) + ["Precision (Std)", "Recall (Std)", "F-measure (Std)", \
                                                        "G-mean (Std)", "AUC (Std)", "Accuracy (Std)"]

        for i, params in enumerate(param_combinations):
            row = list(params) + [round(std_precision[i], 4), round(std_recall[i], 4), round(std_f_measure[i], 4),\
                                                round(std_g_mean[i], 4), round(std_auc[i], 4), round(std_accuracy[i], 4)]
            std_table.add_row(row)

        print("\nStandard Deviation Evaluation Metrics:")
        print(std_table)
    
    # Plot the ROC curve for each configuration
    if(plot_roc):
        for i in range(len(param_combinations)):
            plot_roc_curve(mean_fpr[i], mean_tpr[i], mean_auc[i], list(param_grid.keys()), param_combinations[i])

    if(report_best):
        print("\nBest hyperparameters:")
        for param_name, param_value in best_params.items():
            print(f"{param_name}: {param_value}")
        print(f"Mean G-mean: {round(best_score, 4)}")
        