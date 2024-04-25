import pandas as pd
import pyxlsb
import openpyxl
import time

import glob

print('Работаем.....')

years_list = [2021,2022,2023,2024]  # года которые необходимы
lst_directory = []
print(f"Запущен скрипт сборки заработной платы за {years_list} года")

for year_ in years_list:
    dir_path = f"//sim.local/data/Varsh/OFFICE/CAGROUP/З-п/*{year_}*/ZP-*.xls*"
    for file in glob.glob(dir_path, recursive=True):
        lst_directory.append(file)



stop_list = [' АС', '_', 'Srt', 'Тест', '+', 'моя']

my_list_drop = []
for i in lst_directory:
    counter_ = 0
    for j in range(len(stop_list)):
        if stop_list[j] in i:
            counter_+=1
            
    if counter_ == 0:
        my_list_drop.append(i)
        
        
try:
    df = pd.DataFrame()
    for i in my_list_drop:
        df1 = pd.read_excel(i, sheet_name='Ведом СВОД', header=None).add_prefix("data")
        df1['Истоник'] = i
        df = pd.concat([df, df1])
except:
    print('ошибка вызвана - ', i)

print('минуточку.....')
# переименовываем столбцы
df.rename(columns = {
  'data1': "Таб №",
  'data2': "ФИО",
  'data3': "Должность",
  'data4': "Коэф от ст",
  'data5': "Оклад",
  'data6': "Неопл дн",
  'data7': "Неопл руб",
  'data8': "БЛ дн",
  'data9': "БЛ руб",
  'data10': "Рачетная ЗП",
  'data11': "Вычет-Премия",
  'data12': "Отпуск",
  'data13': "Всего начислено",
  'data14': "Карта ав-с/зп",
  'data15': "Карта премия",
  'data16': "Аванс",
  'data17': "Штрафы",
  'data18': "К выдаче",
  'data19': "Подразделение",
  'data20': "Отдел"
    }, inplace = True )

# список столбцов, которые оставляем
list_colums_necessary = [
  "Таб №",
  'ФИО',
  'Должность',
  'Коэф от ст',
  'Оклад',
  'Неопл дн',
  'Неопл руб',
  'БЛ дн',
  'БЛ руб',
  'Рачетная ЗП',
  'Вычет-Премия',
  'Отпуск',
  'Всего начислено',
  'Карта ав-с/зп',
  'Карта премия',
  'Аванс',
  'Штрафы',
  'К выдаче',
  'Подразделение',
  'Отдел',
  'Истоник'
]

df = df.drop(columns = ['data0'],axis = 1)

print("удаляем и отрезаем лишнее.....")
# удаляем лишние столбцы которых нет в list_colums_necessary

for i in df.columns:                        # пробегаем по списку столбцов df
    if i not in list_colums_necessary:      # если названия столбца нет в списке list_colums_necessary
        
        df = df.drop(columns = [i],axis = 1)    # удаляем столбец
        
# Удаление строк со всеми значениями NaN в каждом столбце
df = df.dropna (how='all')
# Удаление строки со значениями Nan в определенном столбце
df = df.dropna (subset=['ФИО'])
df = df.dropna (subset=['Таб №'])
df = df.dropna (subset=['Отдел'])

df = df.loc[((df['ФИО'] != 'ФИО') & (df['ФИО'] != '0') & (df['ФИО'] != 2) & (df['ФИО'] != '-'))]

# добавляем столбец Автоцентр вытаскивая название из столбца Источник
df['Автоцентр'] = df['Истоник'].apply(lambda x: x[int(x.find("ZP-")+3):-9])

# добавляем столбец Месяц вытаскивая название из столбца Источник
df['Месяц'] = df['Истоник'].apply(lambda x: x[int(x.find(".xls")-4):-7])
#df.head(3)

# добавляем столбец Год вытаскивая название из столбца Источник
df['Год'] = df['Истоник'].apply(lambda x: "20"+x[int(x.find(".xls")-2):-5])
month_correct = lambda x: int(list(x)[1]) if int(list(x)[0])==0 else x
df['Месяц_1'] = df['Месяц'].apply(month_correct)
df['Год'] = df['Год'].astype("Int64")
df['Месяц_1'] = df['Месяц'].astype("Int64")
print("сохраняем.....")
save_file = '//sim.local/data/Varsh/OFFICE/CAGROUP/run_python/task_scheduler/'
print(f'Сохраняем в диреткорию {save_file}')
df.to_csv(f'{save_file}df_ZP_3_years.csv')         # сохраняем в формате csv
df.to_excel(f'{save_file}df_ZP_3_years.xlsx')     # сохраняем в формате xlsx

print('Скрипт закроется через :')
for i in range(5):
    print(5-i)
    time.sleep(.5)