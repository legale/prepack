# prepack
Python excel based data preparation library

## Usage
```
import prepack from prepack as pp
pp.p2c(['one','two','three','four'])
```
## Allowed methods
1. **levenshtein_merge(dfa, dfb, left_on, right_on, limit)** - слияние датафреймов по неполному совпадению через 
самое короткое расстояние Левенштейна. Функция сделана на основе стандартной функции pandas df.merge()
2. **read_excel(filepath)** - читает excel файл в dataframe, изменяя некоторые параметры по умолчанию. 
    1. header=None - не использовать первую строку в качестве заголовка
    2. na_filter=False - отключить обнаружение пустых ячеек и ячеек в русском excel #НД
    3. dtype=str - тип всех ячеек устанавливать строкой.
    
Пример использования: `df = pp.read_excel('2002.xls')`.  

3. **read_excels(filepath)** - тоже самое, только считает все листы в файле, когда нужен доступ
 ко всем листам документа
4. **read_zip(filepath)** - прочитает архив и вернет 2 списка с именами файла и указателями на 
binary file-like object, которые можно сразу считать pandas или просто прочитать через read()
 Пример использования: `names, files = pp.read_zip("raw_data.zip")`
5. **load(filepath)** - прочитать pickle файл
6. **save(obj, filepath)** - записать данные из переменной в pickle файл 
7. **df_filter_and(df, fltr, iloc = False)** - позволяет отфильтровать dataframe без нагромождения 
кода, которое обычно неизбежно. Если установить iloc = True, то можно использовать не имена столбцов, 
а их порядковые номера. Пример использования: 
```
f = pp.df_filter_and(df, {'Наименование показателя': '~isnum', #  не число 
                              'Код по бюджетной классификации ППП': 'isnum', #  число
                              'РзПр': '0100', # тут 0100
                              'ЦСР: 'isblank', # ЦСР # тут пусто
                              'ВР': 'isblank', # ВР # тут пусто
                              'Уточненная сводная бюджетная роспись': 'isnum' #  число
                             })
df = df[f]
```
8. **df_filter_or(df, fltr, iloc = False)** - аналогичная функция. Отличается тем, что тут условия фильтрации
 задаются через ИЛИ, а не через И. Фильтр пропустит строки совпадающие по любому из условий.
9. **p2p(lst)** - печатает список значений в 2 колонки. Главная задача выводить на печать номера и названия
 столбцов.
10. **isnan(var)** - проверяет является ли переменная типом np.nan, отличие от numpy.isnan() в том, что последняя
 возвращает ошибку, если в функцию передано что-то отличное от объекта float или примитива float64.
11. **list_concat(a,b)** - Возьмет непустые значения из списка a и соединит их со значениями из списка b
 через пробел. Делалась для соединения заголовков таблиц, которые обычно записаны сразу в нескольких строках. 
12. **parse_excel(filepath, columns, fltr, header=None)** - считывает и парсит файл excel, возвращает таблицу
 pandas
13. **parse_excels(filelist, columns, fltr, header=None)** - тоже самое, но умеет обрабатывать сразу несколько
 файлов, возвращает 1 большую таблицу, где дополнительным столбцом указано имя файла из которого получены данные.
  Умеет принимать как простой список файлов, так и file-like object. Т.е. можно считать файлы сразу из архива.
