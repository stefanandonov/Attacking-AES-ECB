import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

def calculate_outside_error_for_aes(test_blocks, predicted_blocks):
    """
    :param test_blocks: list of blocks from the test set, where each block is a list of binary numbers 1/0 with
    length 128 :param predicted_blocks: list of blocks which are predicted from the model, where each block is with
    length 128 and each i-th block corresponds to the i-th block from the test set :return: double number,
    in the range 0-1, that represents the outside error from the prediction of AES
    """
    if len(test_blocks) != len(predicted_blocks):
        raise Exception
    sum = 0
    for i in range(0, len(test_blocks)):
        if len(test_blocks[i]) != len(predicted_blocks[i]):
            raise Exception
        for j in range(0, len(test_blocks[i])):
            sum += ((test_blocks[i][j] + predicted_blocks[i][j]) % 2)
    return sum / (len(test_blocks) * len(test_blocks[0]))


def calculate_inside_error_for_aes(test_blocks, predicted_blocks):
    """
    :param test_blocks: list of blocks from the test set, where each block is a list of binary numbers 1/0 with
    length 128 :param predicted_blocks: list of blocks which are predicted from the model, where each block is with
    length 128 and each i-th block corresponds to the i-th block from the test set :return: double number,
    in the range 0-1, that represents the inside error from the prediction of AES
    """
    if len(test_blocks) != len(predicted_blocks):
        raise Exception
    count = 0
    for i in range(0, len(test_blocks)):
        if match(test_blocks[i], predicted_blocks[i]):
            count += 1
    return count / len(test_blocks)


def match(block1, block2):
    """
    :param block1: left block that should be checked if its a complete match with the right block (list of binary numbers)
    :param block2: right block (list of binary numbers)
    :return: boolean that symbolizes that the block1 and block2 are a complete match
    """
    if len(block1) != len(block2):
        raise Exception
    for i in range(0, len(block1)):
        if block1[i] != block2[i]:
            return False
    return True

# plain16 = np.load("FinalPlain16.npy")
# cipher16 = np.load("FinalTrain16.npy")
# print("Data is loaded!")
#
# X = cipher16
# y = plain16
#
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
#
# json_file = open('model16.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model16.h5")
# print("Loaded model from disk")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# loss, accuracy = loaded_model.evaluate(X_test, y_test, batch_size=5000)
# print("Accuracy for 16 byte key is %s: %.2f%%" % (loaded_model.metrics_names[1], accuracy*100))
# print(loss, accuracy)
#
# y_pred = loaded_model.predict(X_test)
# y_pred = np.round(y_pred)
# y_pred = y_pred.astype(int)
#
# np.save("Predicted16.npy", y_pred)
# np.save("Original16", y_test)

# # print(y_pred)
# y_pred = np.round(y_pred)
# # print(y_pred)
# print("Ouside error: ", calculate_outside_error_for_aes(y_test, y_pred))
# print("Inside error: ", calculate_inside_error_for_aes(y_test,y_pred))

# y_pred = loaded_model.predict(X_test)
# y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(y_test, axis=1)
#
#
# print("Precision:", precision_score(y_test, y_pred.round(), average="macro"))
# print("Recall:",recall_score(y_test, y_pred, average="macro", zero_division=1))
# print("F1:", f1_score(y_test, y_pred, average="macro"))
# print("Confusion matrix\n:", confusion_matrix(y_test, y_pred))
# conf_mat=confusion_matrix(y_test, y_pred)
#
#
#
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
# disp = disp.plot()
# plt.show()

# # prediction = loaded_model.predict(X_test)
# # print(prediction)
# batch_size=(len(X_train)+len(y_train))

# plain24 = np.load("FinalPlain24.npy")
# cipher24 = np.load("FinalTrain24.npy")
# print("Data is loaded!")
#
# X = cipher24
# y = plain24
#
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
#
# json_file = open('model24.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model24.h5")
# print("Loaded model from disk")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# loss, accuracy = loaded_model.evaluate(X_test, y_test, batch_size=5000)
# print("Accuracy for 24 byte key is %s: %.2f%%" % (loaded_model.metrics_names[1], accuracy*100))
# print(loss, accuracy)
#
# y_pred = loaded_model.predict(X_test)
# y_pred = np.round(y_pred)
# y_pred = y_pred.astype(int)
#
# np.save("Predicted24.npy", y_pred)
# np.save("Original24", y_test)

# y_pred = loaded_model.predict(X_test)
# # print(y_pred)
# y_pred = np.round(y_pred)
# # print(y_pred)
# print("Ouside error: ", calculate_outside_error_for_aes(y_test, y_pred))
# print("Inside error: ", calculate_inside_error_for_aes(y_test,y_pred))

#
plain32 = np.load("FinalPlain32.npy")
cipher32 = np.load("FinalTrain32.npy")
print("Data is loaded!")

X = cipher32
y = plain32

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

json_file = open('model32.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model32.h5")
print("Loaded model from disk")


# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
loss, accuracy = loaded_model.evaluate(X_test, y_test, batch_size=5000)
print("Accuracy for 32 byte key is %s: %.2f%%" % (loaded_model.metrics_names[1], accuracy*100))
print(loss, accuracy)

y_pred = loaded_model.predict(X_test)
y_pred = np.round(y_pred)
y_pred = y_pred.astype(int)

np.save("Predicted32.npy", y_pred)
np.save("Original32", y_test)

# y_pred = loaded_model.predict(X_test)
# # print(y_pred)
# y_pred = np.round(y_pred)
# y_pred = y_pred.astype(int)
# print(y_pred)
# # print(y_pred)
# outside = calculate_outside_error_for_aes(y_test, y_pred)
# print("Ouside error: ", outside)
# inside = calculate_inside_error_for_aes(y_test,y_pred)
# print("Inside error: ", inside)
# print(classification_report(y_test, y_pred))
#model.predict(X_test, batch_size=32, verbose=1
# def plot_graph_loss(file_name):#, model_name):
#     values = pd.read_table(file_name, sep=',')
#     data = pd.DataFrame()
#     data['epoch'] = list(values['epoch'].get_values() + 1) + list(values['epoch'].get_values() + 1)
#     data['loss name'] = ['training'] * len(values) + ['validation'] * len(values)
#     data['loss'] = list(values['loss'].get_values()) + list(values['val_loss'].get_values())
#     sns.set(style='darkgrid', context='poster', font='Verdana')
#     f, ax = plt.subplots()
#     sns.lineplot(x='epoch', y='loss', hue='loss name', style='loss name', dashes=False, data=data, palette='Set2')
#     ax.set_ylabel('Loss')
#     ax.set_xlabel('Epoch')
#     ax.legend().texts[0].set_text('')
#     #plt.title(model_name)
#     plt.show()