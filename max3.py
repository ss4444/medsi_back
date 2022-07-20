from heapq import nlargest


def three(class_names, predict):
    max_class_names = []
    max_predict = nlargest(3, predict)
    for i in range(len(max_predict)):
        index = predict.index(max_predict[i])
        max_class_names.append(class_names[index])
    return max_class_names, max_predict
