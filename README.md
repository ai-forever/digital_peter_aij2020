# AI Journey 2020 Digital Peter

Соревнование по распознаванию древних текстов, написанных рукой Петра Великого.

### Описание задачи и данных

Участникам предлагается построчно распознавать рукописный текст Петра I.

Train выборку можно скачать [тут](https://drive.google.com/file/d/1kDmRCl692k6s9kQnNryq5ByAaHZX2uEw/view?usp=sharing).

Внутри находятся 2 папки:  images и words. В папке images лежат jpg-файлы с вырезанными строками из документов Петра Великого, в папке words - txt-файлы (транскрибированные версии). Маппинг осуществляется по названию.

Например, 

оригинал (1_1_10.jpg):
<p align="center">
  <img src="pics/1_1_10.jpg" width="70%">
</p>

его перевод (1_1_10.txt):
```bash
                                    зело многа в гафѣ i непърестано выхо
```

### Бейзлайн

Ноутбук с бейзлайном задачи:
```bash
baseline.ipynb
```

### Описание метрик

В лидерборде будут учитываться следующие метрики качества распознавания (на тестовой выборке)

* **CER** - Character Error Rate 

<p align="center">
  <img src="pics/CER.png" width="30%">
</p>

Здесь <img src="https://render.githubusercontent.com/render/math?math=\text{dist}_c"> - это расстояние Левенштейна, посчитанное для токенов-символов (включая пробел), <img src="https://render.githubusercontent.com/render/math?math=\text{len}_c"> - длина строки в символах.

* **WER** - Word Error Rate

<p align="center">
  <img src="pics/WER.png" width="30%">
</p>

Здесь <img src="https://render.githubusercontent.com/render/math?math=\text{dist}_w"> - это расстояние Левенштейна, посчитанное для токенов-слов, <img src="https://render.githubusercontent.com/render/math?math=\text{len}_w"> - длина строки в словах.

* **Sentence Accuracy** - отношение количества полностью совпавших строк (учитывая пробелы) к количеству строк в выборке.

<p align="center">
  <img src="pics/SentenceAccuracy.png" width="40%">
</p>

В этой формуле используется скобка Айверсона:
<p align="center">
  <img src="pics/IversonBracket.png" width="20%">
</p>

В формулах выше <img src="https://render.githubusercontent.com/render/math?math=n"> - размер тестовой выборки, <img src="https://render.githubusercontent.com/render/math?math=\text{pred}_i"> - это строка из символов, которую распознала модель на <img src="https://render.githubusercontent.com/render/math?math=i">-ом изображении, а <img src="https://render.githubusercontent.com/render/math?math=\text{true}_i"> - это истинный перевод <img src="https://render.githubusercontent.com/render/math?math=i">-ого изображения, произведенный экспертом.


Про метрики дополнительно можно прочитать [тут](https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates). Методику подсчета метрик можно изучить подробнее в скрипте evaluate.py. Он принимает на вход два параметра - pred.txt и true.txt. Это файлы со строками предсказаний и со строками реальных ответов соответственно.

Главная метрика, по которой сортируется лидерборд, - CER (меньше - лучше). В случае совпадения CER у двух или более участников, сортировка для них будет вестись по WER (меньше - лучше). Если и CER, и WER совпадают, - смотрим на Sentence Accuracy (больше - лучше).


### Отправка решения
TBD
