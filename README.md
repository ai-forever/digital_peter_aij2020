# AI Journey 2020 Digital Peter

Соревнование по распознаванию древних текстов, написанных рукой Петра Великого.

### Описание задачи и данных

Участникам предлагается построчно распознавать рукописный текст Петра I.

Train выборку можно скачать [тут](https://drive.google.com/file/d/1kDmRCl692k6s9kQnNryq5ByAaHZX2uEw/view?usp=sharing).

Внутри находятся 2 папки:  images и words. В папке images лежат .jpg файлы с вырезанными строками из документов Петра Великого, в папке words - .txt файлы (транскрибированные версии). Маппинг осуществляется по названию.

Например, 

оригинал (1_1_10.jpg):
<p align="center">
  <img src="pics/1_1_10.jpg" width="70%">
</p>

его перевод (1_1_10.txt):
```bash
-------------------------- зело многа в гафѣ i непърестано выхо
```

### Бейзлайн

Ноутбук с бейзлайном задачи:
```bash
baseline.ipynb
```

### Описание метрик

В лидерборде будут учитываться следующие метрики качества распознавания (на test выборке):

* **CER** - Character Error Rate 

<p align="center">
  <img src="pics/CER.png" width="40%">
</p>

Здесь <img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">


* **WER** - Word Error Rate

<p align="center">
  <img src="pics/WER.png" width="40%">
</p>

* **Sentence Accuracy** - отношение количества полностью совпавших строк (учиитывая пробелы) к количеству строк в тестовой выборке.

<p align="center">
  <img src="pics/SentenceAccuracy.png" width="50%">
</p>


В формулах для CER и WER dist - это расстояние Левенштейна. Только для CER токеном является символ (dist_c), а для WER токеном является слово (dist_w). Методику подсчета метрик можно изучить подробнее в скрипте evaluate.py. Он принимает на вход два параметра - pred.txt и true.txt. Это файлы со строками предсказаний и со строками реальных ответов соответственно.


Главная метрика, по которой сортируется лидерборд, - <img src="https://render.githubusercontent.com/render/math?math=\text{CER}"> (меньше - лучше). В случае совпадения CER у двух или более участников, сортировка для них будет вестись по WER (меньше - лучше). Если и CER, и WER совпадают, - смотрим на Sentence Accuracy (больше - лучше).

Про метрики дополнительно можно прочитать вот здесь https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates.
