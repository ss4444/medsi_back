def three(class_names, predict):
    max_class_names = []
    max_predict = []
    for g in range(3):
        for i in range(len(predict)):
            if predict[i] == max(predict):
                max_predict.append(max(predict))
                max_class_names.append(class_names[i])
        predict.remove(max(predict))
        class_names.remove(max_class_names[-1])
    return max_class_names, max_predict