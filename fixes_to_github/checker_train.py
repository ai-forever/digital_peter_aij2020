# -*- coding: utf-8 -*-

import sys
import os
import io
import re
import numpy as np
from copy import deepcopy


def checker(word_path):
    """
    Fix and rewrite sentences in train dataset. Which mistakes of annotation were fixed is clear from the code

    :param word_path: path to the folder with ground truth sentences, type str

    :return (num_corr, num), type tuple
        num_corr: number of corrected files in the folder with ground truth sentences, type int
        num:      number of all files in the folder with ground truth sentences, type int
    """

    filenames = sorted(os.listdir(word_path))
    num = len(filenames) 
    num_corr = 0  # number of corrected files

    for fn in filenames:
        with io.open(word_path+'/'+fn, 'r',  encoding='utf8') as file:
            data = file.read()
        
        cntr = 0 # number (in some sense) of corrections inside the current file

        # to have ability to print previous version of a string
        data_old = deepcopy(data)

        if fn == '377_2_16.txt':
            data = 'шат сей пробы а с протчими лѣсами (+ поспѣ'
            cntr += 1
        if fn == '446_8_5.txt':
            data = 'къголца також вскорѣ поспеют (+'
            cntr += 1            
        if fn == '80_35_31.txt':
            data = 'но хлѣбъ сол мяса са (+ давано было'
            cntr += 1
        if fn == '302_53_3.txt':
            data = 'как вас бог наставит i лутче можете дѣлат (+'
            cntr += 1
        if fn == '380_1_7.txt':
            data = 'были чрез переволочну близ шведоф [как'
            cntr += 1
        if fn == '382_11_19.txt':
            data = 'нынѣ лесофъ согънат а паче кривуль i досокъ на по (+'
            cntr += 1
        if fn == '182_4_5.txt':
            data = 'прибавит маленкие 2 фланки (+ чтоб на каждой по двѣ'
            cntr += 1
        if fn == '362_4_14.txt':
            data = 'рой нынѣ двоеполубной дѣлают] i ситадел (+ [для славы'
            cntr += 1
        if fn == '460_9_9.txt':
            data = 'щего (+ iже воздастъ комуж'
            cntr += 1
        if fn == '269_12_14.txt':
            data = 'вѣтоф ближе (+ та'
            cntr += 1
        if fn == '337_46_6.txt':
            data = 'ном (+ [хотя не для держания]'
            cntr += 1
        if fn == '368_8_4.txt':
            data = 'божих лѣсоф карабля на три (+ [60'
            cntr += 1
        if fn == '347_56_9.txt':
            data = 'тират (× чего всегда i от в в неот'
            cntr += 1
        if fn == '446_4_5.txt':
            data = 'в свѣю отпустят (+'
            cntr += 1
        if fn == '284_3_35.txt':
            data = 'пъримет то оной разорят (+'
            cntr += 1
        if fn == '376_18_26.txt':
            data = 'кѣ присылат i i о том поговори з дру (+'
            cntr += 1
        if fn == '106_1_22.txt':
            data = 'а для того чтоб хан повѣрил ⊕ дан ему пасъ за го'
            cntr += 1
        if fn == '445_9_25.txt':
            data = '⊕ оставя в печерской'
            cntr += 1
        if fn == '368_2_5.txt':
            data = 'поставит для пробы впрет ⊕ почему станут ставит'
            cntr += 1
        if fn == '343_46_32.txt':
            data = '⊕ для чего удобнѣе всѣх мѣстъ'
            cntr += 1
        if fn == '377_2_10.txt':
            data = 'впрѣд ⊕'
            cntr += 1
        if fn == '74_24_12.txt':
            data = '⊕ i у его кс поттвер'
            cntr += 1
        if fn == '335_38_29.txt':
            data = '⊕ i мню что то iли друг'
            cntr += 1
        if fn == '413_14_12.txt':
            data = 'нѣсколко времени ⊕ потом уступит'
            cntr += 1
        if fn == '326_26_12.txt':
            data = '⊕ сия первоя грамо'
            cntr += 1
        if fn == '446_7_5.txt':
            data = '⊕ i вашим прислан'
            cntr += 1
        if fn == '85_53_0.txt':
            data = '⊕ писем отправили с писмом к вам что сей трактат принят'
            cntr += 1
        if fn == '220_19_4.txt':
            data = '⊕ а оной карабль для осмотру'
            cntr += 1
        if fn == '413_14_34.txt':
            data = '⊕ дабы утомѣл'
            cntr += 1
        if fn == '398_3_0.txt':
            data = '⊕ також i картины [как слышим]'
            cntr += 1
        if fn == '335_38_18.txt':
            data = 'нея во въ флотѣ состоiт ⊕'
            cntr += 1
        if fn == '268_19_0.txt':
            data = '⊕ курфиртъ зело склонен явился i совѣ'
            cntr += 1
        if fn == '85_56_1.txt':
            data = '⊕ на трактат учиненно въ ны 1713 в адриано'
            cntr += 1
        if fn == '325_25_16.txt':
            data = 'дению есть ⊕ [i чтоб не iзволил слабѣт в при'
            cntr += 1
        if fn == '445_9_17.txt':
            data = 'ва междо двух крѣпостѣй ⊕ к лы'
            cntr += 1
        if fn == '226_15_13.txt':
            data = 'тел стоял а когда въ морѣ вкроетца тог'
            cntr += 1
        if fn == '47_20_5.txt':
            data = 'заваеванное i въ том говорит отданием лифляндов'
            cntr += 1
        if fn == '109_3_1.txt':
            data = 'p s для бога старайтес прислат'
            cntr += 1
        if fn == '110_14_5.txt':
            data = 'p s получили мы третево дни въве'
            cntr += 1
        if fn == '19_4_7.txt':
            data = 'p s получили мы писмо от гетмана х князю'
            cntr += 1
        if fn == '356_5_0.txt':
            data = 'sire +'
            cntr += 1
        if fn == '41_13_4.txt':
            data = 'p s найпаче всего смотрѣт дабы турок'
            cntr += 1
        if fn == '58_9_6.txt':
            data = 'p s'
            cntr += 1
        if fn == '85_55_1.txt':
            data = 'p s понеже безотложно бытие мое'
            cntr += 1
        if fn == '14_29_10.txt':
            data = 'p s карабль наш сваятого петра сего'
            cntr += 1
        if fn == '17_7_10.txt':
            data = 'p s поздравляем сим днем'
            cntr += 1
        if fn == '8_2_11.txt':
            data = 'p s отпиши ко мнѣ х которому'
            cntr += 1

        data, n_subs = re.subn('c', 'с', data)
        # n_subs - number of substitutions
        cntr += n_subs

        data, n_subs = re.subn('і', 'i', data)
        cntr += n_subs

        data, n_subs = re.subn('ҍ', 'ѣ', data)
        cntr += n_subs

        # we replace multiple spaces to one
        flag = False
        # 500 is an upper bound (with a large margin:) ) of the maximum number of characters in string
        for k in range(2,500):
            if ' ' * k in data:
                flag = True
        if flag:
            data = re.subn(r'\s+', ' ', data)[0]
            cntr += 1
        
        if fn == '380_1_8.txt':
            data = data.replace(')',']')
            cntr += 1
        if fn == '188_4_11.txt':
            data = data.split()
            data.remove('+)')
            data.insert(1,'⊕')
            data = ' '.join(data)
            cntr += 1

        num_corr += np.sign(cntr)

        if cntr > 0:
            with open(word_path+'/'+fn, 'w') as file: 
                file.write(data)
            print('[' + fn + ']: ' + data_old + ' --> ' + data)

    return num_corr, num


if __name__ == '__main__':
    path_to_words = sys.argv[1]
    num_corr, num = checker(path_to_words)
    print('\nSTATISTICS')
    print('Number of corrected files = ' + str(num_corr))
    print('Total number of files = ' + str(num))
    print('Percentage of corrected files = ' + str(np.round(num_corr/num * 100, 2)) + '%')
