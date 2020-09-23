import sys
import editdistance

def evaluate():
    pred_path = sys.argv[1]
    true_path = sys.argv[2]


    with open('pred.txt') as f:
        prediction = f.readlines()
    predictions = [x.rstrip('\n') for x in prediction] 

    with open('true.txt') as f:
        true_text = f.readlines()

    true_text = [x.rstrip('\n') for x in true_text] 


    numCharErr = 0
    numCharTotal = 0
    numStringOK = 0
    numStringTotal = 0

    word_eds, word_true_lens = [], []

    print('Ground truth -> Recognized')	
    for i in range(len(predictions)):
        pred = predictions[i]
        true = true_text[i]

        numStringOK += 1 if true == pred else 0
        numStringTotal += 1
        dist = editdistance.eval(pred, true)
        numCharErr += dist
        numCharTotal += len(true)

        pred_words = pred.split()
        true_words = true.split()
        word_eds.append(editdistance.eval(pred_words, true_words))
        word_true_lens.append(len(true_words))

        print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + true + '"', '->', '"' + pred + '"')

    charErrorRate = numCharErr / numCharTotal
    wordErrorRate = sum(word_eds) / sum(word_true_lens) 
    stringAccuracy = numStringOK / numStringTotal
    print('Character error rate: %f%%. Word error rate: %f%%. String accuracy: %f%%.' % \
          (charErrorRate*100.0,wordErrorRate*100.0, stringAccuracy*100.0))

if __name__ == "__main__":
    evaluate()