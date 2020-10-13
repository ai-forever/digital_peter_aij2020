<p align="center">
  <img src="pics/AIJ_Logo.png" width="100%">
</p>

# Digital Peter: распознавание рукописей Петра I

The English version of this document is [here](https://github.com/sberbank-ai/digital_peter_aij2020/blob/master/README.en.md)

Соревнование по распознаванию древних текстов, написанных рукой Петра Великого.

### Описание задачи и данных

Участникам предлагается построчно распознавать рукописный текст Петра I.

Развернутое описание задачи (с погружением в проблематику) можно прочитать в [```desc/detailed_description_of_the_task_ru.pdf```](https://github.com/sberbank-ai/digital_peter_aij2020/blob/master/desc/detailed_description_of_the_task_ru.pdf)

Train выборку можно скачать [тут](https://storage.yandexcloud.net/datasouls-ods/materials/46b7bb85/datasets.zip). Выборка была подготовлена совместно с рабочей группой, состоящей из научных сотрудников СПбИИ РАН - специалистов по истории Петровской эпохи, а также палеографии и археографии. Большую помощь оказали Росархив и РГАДА, которые предоставили цифровые копии автографов.

Внутри находятся 2 папки:  ```images``` и ```words```. В папке ```images``` лежат jpg-файлы с вырезанными строками из документов Петра Великого, в папке ```words``` - txt-файлы (транскрибированные версии jpg-файлов). Маппинг осуществляется по названию.

Например, 

оригинал (1_1_10.jpg):
<p align="center">
  <img src="pics/1_1_10.jpg" width="70%">
</p>

его перевод (1_1_10.txt):
```bash
                                  зело многа в гафѣ i непърестано выхо
```

Имена файлов имеют следующий формат `x_y_z`, где `x` - номер серии (серия - некоторый набор страниц с текстом), `y` - номер страницы, `z` - номер строки на этой странице. 
Абсолютные значения `x`, `y`, `z` не несут никакого смысла (это внутренние номера). Важна лишь последовательность `z` при фиксированном `x_y`. Например, в файлах 
```
  987_65_10.jpg
  987_65_11.jpg
  987_65_12.jpg
  987_65_13.jpg
  987_65_14.jpg
```
точно находятся 5 идущих подряд строк. Этот факт можно использовать дополнительно. 

Названия файлов в тестовой выборке имеют такую же структуру.

Подавляющее большинство строк написаны рукой Петра Великого в период с 1709 по 1713 годы (есть строки, написанные в 1704, 1707 и 1708 годах, но их не более 150; эти строки попали как в [train](https://storage.yandexcloud.net/datasouls-ods/materials/46b7bb85/datasets.zip), так и в test).


### Бейзлайн

Ноутбук с бейзлайном задачи:
[```baseline.ipynb```](https://github.com/sberbank-ai/digital_peter_aij2020/blob/master/baseline.ipynb)

Для распознавания текста (в бейзлайне) используется следующая архитектура:

<p align="center">
  <img src="pics/ArchitectureNN.jpg" width="60%">
</p>


### Описание метрик

В лидерборде будут учитываться следующие метрики качества распознавания (на тестовой выборке)

* **CER** - Character Error Rate 

<p align="center">
  <img src="pics/CER.png" width="30%">
</p>

Здесь <img src="https://render.githubusercontent.com/render/math?math=\text{dist}_c"> - это расстояние Левенштейна, посчитанное для токенов-символов (включая пробелы), <img src="https://render.githubusercontent.com/render/math?math=\text{len}_c"> - длина строки в символах.

* **WER** - Word Error Rate

<p align="center">
  <img src="pics/WER.png" width="30%">
</p>

Здесь <img src="https://render.githubusercontent.com/render/math?math=\text{dist}_w"> - это расстояние Левенштейна, посчитанное для токенов-слов, <img src="https://render.githubusercontent.com/render/math?math=\text{len}_w"> - длина строки в словах.

* **Sentence Accuracy** - отношение количества полностью совпавших строк к количеству строк в выборке.

<p align="center">
  <img src="pics/SentenceAccuracy.png" width="40%">
</p>

В этой формуле используется скобка Айверсона:
<p align="center">
  <img src="pics/IversonBracket.png" width="20%">
</p>

В формулах выше <img src="https://render.githubusercontent.com/render/math?math=n"> - размер тестовой выборки, <img src="https://render.githubusercontent.com/render/math?math=\text{pred}_i"> - это строка из символов, которую распознала модель на <img src="https://render.githubusercontent.com/render/math?math=i">-ом изображении, а <img src="https://render.githubusercontent.com/render/math?math=\text{true}_i"> - это истинный перевод <img src="https://render.githubusercontent.com/render/math?math=i">-ого изображения, произведенный экспертом.


Про метрики дополнительно можно прочитать [тут](https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates). 

Методику подсчета метрик можно изучить подробнее в скрипте [```eval/evaluate.py```](https://github.com/sberbank-ai/digital_peter_aij2020/blob/master/eval/evaluate.py). Он принимает на вход два параметра - [```eval/pred_dir```](https://github.com/sberbank-ai/digital_peter_aij2020/tree/master/eval/pred_dir) и [```eval/true_dir```](https://github.com/sberbank-ai/digital_peter_aij2020/tree/master/eval/true_dir). В папке [```eval/true_dir```](https://github.com/sberbank-ai/digital_peter_aij2020/tree/master/eval/true_dir) должны находиться txt-файлы с истинным переводом строк (структура как в папке ```words```), в папке [```eval/pred_dir```](https://github.com/sberbank-ai/digital_peter_aij2020/tree/master/eval/pred_dir) - txt-файлы, содержащие распознанные (моделью) строки. Маппинг опять же осуществляется по названию, поэтому списки названий файлов в папках [```eval/true_dir```](https://github.com/sberbank-ai/digital_peter_aij2020/tree/master/eval/true_dir) и [```eval/pred_dir```](https://github.com/sberbank-ai/digital_peter_aij2020/tree/master/eval/pred_dir) **должны полностью совпадать**!

Качество можно посчитать следующей командой (вызванной из папки [```eval```](https://github.com/sberbank-ai/digital_peter_aij2020/tree/master/eval)):

```bash
python evaluate.py pred_dir true_dir
```

Результат отображается в следующем виде:
```bash
Ground truth -> Recognized
[ERR:3] "Это соревнование посвящено" -> "Эт срвнование посвящено"
[ERR:3] "распознаванию строк из рукописей" -> "распознаваниюстр ок из рукписей"
[ERR:2] "Петра I" -> "Птра 1"
[OK] "Удачи!" -> "Удачи!"
Character error rate: 11.267606%
Word error rate: 70.000000%
String accuracy: 25.000000%
```

Главная метрика, по которой сортируется лидерборд, - **CER**, %, (меньше - лучше). В случае совпадения **CER** у двух или более участников, сортировка для них будет вестись по **WER**, %, (меньше - лучше). Если и **CER**, и **WER** совпадают, - смотрим на **Sentence Accuracy**, %, (больше - лучше). Следующий показатель - время работы модели на тестовой выборке, **Time**, sec., (меньше - лучше). Если все метрики сопадают, тогда первым будет решение, загруженное раньше по времени (если и тут все совпадает, то сортируем по алфавиту по названиям команд).

Последняя версия модели (см. [```baseline.ipynb```](https://github.com/sberbank-ai/digital_peter_aij2020/blob/master/baseline.ipynb)) имеет следующие значения метрик качества, посчитанных на public-части тестовой выборки:
```bash
CER = 10.526%
WER = 44.432%
String Accuracy = 21.662%
Time = 60 sec
```

### Формат решения

В качестве решений принимается алгоритм (код + необходимые файлы) и описание точки запуска в виде одного архива. В корне архива с решением должен лежать файл `metadata.json` со структурой:

```
{
   "image": "<docker image>",
   "entrypoint": "<entry point or sh script>"
}
```

Например:
```
{
   "image": "odsai/python-gpu",
   "entrypoint": "python predict.py"
}
```

Данные следует читать из папки `/data`, предсказания писать в `/output`. Для каждой картинки из папки `/data` 
вида `<image_name>.jpg` нужно записать в `/output` `<image_name>.txt` с распознанным текстом.

Решение запускается в Docker контейнере. Вы можете воспользоваться готовым образом https://hub.docker.com/r/odsai/python-gpu. В нем предустановлены CUDA 10.1, CUDNN 7.6 и актуальные версии Python библиотек. При желании вы можете использовать свой образ, выложив его на https://hub.docker.com.

Доступные ресурсы:
- 8 ядер CPU 
- 94Gb RAM
- Видеокарта NVidia Tesla V100

Ограничения:
- 5Gb памяти для рабочей директории
- 5Gb на архив с решением
- 10 минут на работу решения

Пример можно посмотреть в [`submit_example`](https://github.com/sberbank-ai/digital_peter_aij2020/tree/master/submit_example)
