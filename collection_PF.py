import pandas as pd
import glob
import time

years_list = [2021, 2022, 2023]  # года которые необходимы
lst_directory = []

print(f"Запущен скрипт сборки PF за {years_list} года")

for year_ in years_list:
    dir_path = f"//sim.local/data/Varsh/OFFICE/CAGROUP/*{year_}*/P-F_????.xlsx*"
    for file in glob.glob(dir_path, recursive=True):
        lst_directory.append(file)
        
my_list_drop = []
for i in lst_directory:
    if '_1' not in i and '_12' not in i and '_SLS' not in i and 'fin' not in i: # в названии файлов не должно присутствовать упоминаний
        my_list_drop.append(i)

# указываем нудные нам листы в PF для сбора данных
list_autocentre = ['cnt','svr','yzs','yzk','yzm','ychr','ykz','yti','ym','yh','yj','yv','yr','ryb','sar']

# проверка на повторы
if len(set(list_autocentre)) == len(list_autocentre):
    print("НЕТ ПОВТОРОВ")
else:
    print("ОШИБКА - ЕСТЬ ПОВТОРЫ НАЗВАНИЯ ЛИСТОВ")
spav_month = {3:"Январь", 4:"Январь", 5:"Февраль", 6:"Февраль", 7:"Март", 8:"Март", 11:"Апрель", 12:"Апрель", 
              13:"Май", 14:"Май", 15:"Июнь", 16:"Июнь", 19:"Июль", 20:"Июль", 21:"Август", 22:"Август", 23:"Сентябрь", 24:"Сентябрь",
              27:"Октябрь", 28:"Октябрь", 29:"Ноябрь", 30:"Ноябрь", 31:"Декабрь", 32:"Декабрь"}

spav_pf = {3:"План", 4:"Факт", 5:"План", 6:"Факт", 7:"План", 8:"Факт", 11:"План", 12:"Факт", 13:"План", 
           14:"Факт", 15:"План", 16:"Факт", 19:"План", 20:"Факт", 21:"План", 22:"Факт", 23:"План", 24:"Факт",
           27:"План", 28:"Факт", 29:"План", 30:"Факт", 31:"План", 32:"Факт"}
# напишем функции
funk_month = lambda x: spav_month.get(x) if x in spav_month else "Неизвестно"
funk_pf = lambda x: spav_pf.get(x) if x in spav_pf else "Неизвестно"
print('Основная обработка файла.....')
df_itog = pd.DataFrame()
for years in my_list_drop:

    df_all = pd.DataFrame()
    for autocentre in list_autocentre:
        
        try:
            df_pf = pd.read_excel(years, sheet_name=autocentre, header=None)

            df_plan = df_pf.melt(id_vars=[0,2], 
                        value_vars=[3,5,7,11,13,15,19,21,23,27,29,31]) # планы
            df_plan['месяц_п'] = df_plan['variable'].apply(funk_month)
            df_plan['автоцентр_п'] = autocentre
            df_plan['источник_п'] = years
            df_plan['год_п'] = df_plan['источник_п'].apply(lambda x: "20"+x[int(x.find(".xls")-2):-5])
            df_plan = df_plan.rename(columns={df_plan.columns[0]: 'индекс_п', df_plan.columns[1]: 'наименование_п',
                                            df_plan.columns[2]: 'удалить_1', df_plan.columns[3]: 'план'})
            df_plan = df_plan[~((df_plan.duplicated(['индекс_п','план']))&(df_plan.индекс_п.isin(['3120н','3130н','3140н','2580н'])))] # удаляем дубликаты БП план ток по ней косяк
            
            df_fakt = df_pf.melt(id_vars=[0,2], 
                        value_vars=[4,6,8,12,14,16,20,22,24,28,30,32]) # факты
            df_fakt['месяц_ф'] = df_fakt['variable'].apply(funk_month)
            df_fakt['автоцентр_ф'] = autocentre
            df_fakt['источник_ф'] = years
            df_fakt['год_ф'] = df_fakt['источник_ф'].apply(lambda x: "20"+x[int(x.find(".xls")-2):-5])
            df_fakt = df_fakt.rename(columns={df_fakt.columns[0]: 'индекс_ф', df_fakt.columns[1]: 'наименование_ф',
                                            df_fakt.columns[2]: 'удалить_2', df_fakt.columns[3]: 'факт'})
            df_fakt = df_fakt[~((df_fakt.duplicated(['индекс_ф','факт']))&(df_fakt.индекс_ф.isin(['3120н','3130н','3140н','2580н'])))] # удаляем дубликаты БП факт ток по ней косяк
            
            df_AC = pd.concat([df_plan,df_fakt], axis=1)
            df_all = pd.concat([df_all,df_AC])
            
        except:
            print(years, "НЕ УДАЛОСЬ НАЙТИ ЛИСТ ",autocentre)
            
    df_itog = pd.concat([df_itog, df_all])
    
try:
    
    df_itog = df_itog.drop(['удалить_1', 'удалить_2','индекс_ф','наименование_ф','месяц_п', 'автоцентр_п', 'источник_п', 'год_п'], axis=1)
    df_itog
except:
    print("Уже применено - лучше пересчитать ВСЕ заново")
print('Ожидайте.........')
df_itog = df_itog.rename(columns={'индекс_п': 'индекс', 'наименование_п' :"наименование", 'месяц_ф':'месяц','автоцентр_ф':'автоцентр', 'источник_ф':'источник', "год_ф":'год'})
df_itog = df_itog[df_itog['факт'] != '[---]']

# список цифровизации месяца
month_digit = {'Январь':1, 'Февраль':2, 'Март':3, 'Апрель':4, 'Май':5, 'Июнь':6, 'Июль':7, 'Август':8, 'Сентябрь':9, 'Октябрь':10, 'Ноябрь':11, 'Декабрь':12}
# функция цифровизации месяца
funk_month_digit = lambda x: month_digit.get(x) if x in month_digit else "Неизвестно"

df_itog['месяц_digit'] = df_itog['месяц'].apply(funk_month_digit)
df_itog['год'] = df_itog['год'].astype("Int64")
df_itog = df_itog[df_itog['месяц_digit'] != 'Неизвестно']

# директория куда сохраняем
save_file = '//sim.local/data/Varsh/OFFICE/CAGROUP/run_python/task_scheduler/'
print(f'Сохраняем файл в диреторию .. {save_file}')
df_itog.to_excel(f'{save_file}df_PF_3_years_вертикаль.xlsx')
df_itog.to_csv(f'{save_file}df_PF_3_years_вертикаль.csv')

print('Файл закроется через :')
for i in range(5):
    print(5-i)
    time.sleep(.5)
    
