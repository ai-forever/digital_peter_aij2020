# -*- coding: utf-8 -*-

import sys
import editdistance

#Это скрипт который посчитает 3 метрики качества, по которым оценивается решение
def evaluate():
    #Аргументы - названия файлов с предсказаниями и правильными ответами.
    #В true записаны построчно правильные ответы. В pred - соответствующие предсказания
    pred_path = sys.argv[1]
    true_path = sys.argv[2]


    with open(pred_path) as f:
        prediction = f.readlines()
    predictions = [x.rstrip('\n') for x in prediction] 

    with open(true_path) as f:
        true_text = f.readlines()

    true_text = [x.rstrip('\n') for x in true_text] 

    #По сути, мы в цикле пробегаемся по предсказаниям, считаем расстояние Левенштейна, а затем делим сумму расстояний на сумму длин всех правильных ответов
    def cer():
        numCharErr = 0
        numCharTotal = 0
        
        for i in range(len(predictions)):
            pred = predictions[i]
            true = true_text[i]
            dist = editdistance.eval(pred, true)
            numCharErr += dist
            numCharTotal += len(true)
        charErrorRate = numCharErr / numCharTotal
        return charErrorRate*100
        
    #Аналогичный подход, как и для CER. Только объектом является слово, а не символ. Соответственно в подсчете расстояния участвует два массива - pred и true, в массивах содержатся слова
    def wer():
        word_eds, word_true_lens = [], []
        for i in range(len(predictions)):
            pred = predictions[i]
            true = true_text[i]

            pred_words = pred.split()
            true_words = true.split()
            word_eds.append(editdistance.eval(pred_words, true_words))
            word_true_lens.append(len(true_words))
    
        wordErrorRate = sum(word_eds) / sum(word_true_lens)
        return wordErrorRate*100
        
    #Наиболее простая метрика, которая считает количество полных совпадений предложений (pred и true) и делит его на общий размер теста
    def string_acc():
        numStringOK = 0
        numStringTotal = 0

        for i in range(len(predictions)):
            pred = predictions[i]
            true = true_text[i]

            numStringOK += 1 if true == pred else 0
            numStringTotal += 1
        
        stringAccuracy = numStringOK / numStringTotal
        
        return stringAccuracy*100
    
    
    #Вызов функций и подсчет метрик
    charErrorRate = cer()
    wordErrorRate = wer()
    stringAccuracy = string_acc()
    
    #Выведем построчно правильные ответы и предсказания
    print('Ground truth -> Recognized')
    for i in range(len(predictions)):
        pred = predictions[i]
        true = true_text[i]

        dist = editdistance.eval(pred, true)

        print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + true + '"', '->', '"' + pred + '"')
    
    print('Character error rate: %f%%' % charErrorRate)
    print('Word error rate: %f%%' % wordErrorRate)
    print('String accuracy: %f%%' % stringAccuracy)
    

if __name__ == "__main__":
    evaluate()
