# digital_peter_aij2020

baseline.ipynb - Ноутбук с бейзлайном задачи.

Данные лежат вот [тут](https://drive.google.com/file/d/1kDmRCl692k6s9kQnNryq5ByAaHZX2uEw/view?usp=sharing).

### Краткое описание задачи и метрики

Участникам предлагается построчно распознавать рукописный текст Петра 1.

В качестве метрик качества мы используем следующие:
* CER - mean character error rate 

\begin{equation}
\text{CER} = \frac{\sum\limits_{i=1}^n \text{dist}_c (\text{pred}_i,\text{true}_i)}{\sum\limits_{i=1}^n \text{len} (\text{true}_i)}
\end{equation}

* WER - mean word error rate (среднее по test-выборке строк);
* Sentence Accuracy - число полностью совпавших строк в test / общее число строк в test.

Главная метрика, по которой сортируется лидерборд, - MCER (меньше - лучше). В случае совпадения MCER у двух или более участников, сортировка для них будет вестись по MWER (меньше - лучше). Если и MCER, и MWER совпадают, - смотрим на Sentence Accuracy (больше - лучше).

Про метрики дополнительно можно прочитать вот здесь https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates.
