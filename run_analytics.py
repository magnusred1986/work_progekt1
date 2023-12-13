import os
import pandas as pd
import warnings
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from fpdf import FPDF
from decimal import Decimal
import dataframe_image as dfi
import df2img
from datetime import datetime
pd.options.display.float_format = "{:.2f}".format


warnings.filterwarnings('ignore')  # игнорируем предупреждения
pd.options.display.max_colwidth = 100  # увеличить максимальную ширину столбца
pd.set_option('display.max_columns', None)  # макс кол-во отображ столбц

# считываем данные переданные пользователем в панеле управления
with open("glb_file.txt", "r", encoding='utf-8') as file:
    x = file.readlines()
    x = [i.replace('\n', '') for i in x]
    name_ac, month_pf, year_pf, save_string = x

spravka_month_reversed = {'Январь': 1, 'Февраль': 2, 'Март': 3, 'Апрель': 4,
                 'Май': 5, 'Июнь': 6, 'Июль': 7, 'Август': 8,
                 'Сентябрь': 9, 'Октябрь': 10, 'Ноябрь': 11, 'Декабрь': 12}

month_pf = spravka_month_reversed.get(month_pf)
save_string = save_string.replace("'",'')

current_datetime = datetime.now()
time_str_config = (f"{current_datetime.day}_"
                   f"{current_datetime.month}_"
                   f"{current_datetime.year}_t_"
                   f"{current_datetime.hour}_"
                   f"{current_datetime.minute}")

# БЛОК КОДА ПОСЛЕ КОРРЕКТИРОВКИ В ipynb вставляем отсюда
# подключение к БД
def connect_bd():
    """Функция подключения к исходным bd: pf, zp, шрифты_fpdf

    Returns:
        _type_: df = PF, df = ZP, ссылка на папку со шрифтами
    """
    all_test = 0
    bd_pf_list = ['//sim.local/data/Varsh/OFFICE/CAGROUP/run_python/task_scheduler/df_PF_3_years_вертикаль.csv',
                  '/Users/sergey_krutko/vscode_projects/тестовая/df_PF_3_years_вертикаль.csv']
    bd_zp_list = ['//sim.local/data/Varsh/OFFICE/CAGROUP/run_python/task_scheduler/df_ZP_3_years.csv',
                  '/Users/sergey_krutko/vscode_projects/тестовая/df_ZP_3_years.csv']
    bd_font_dir_list = ['//sim.local/data/Varsh/OFFICE/CAGROUP/run_python/font/',
                        '/Users/sergey_krutko/vscode_projects/тестовая/font/',
                        '/font/', '//sim.local/data/Varsh/OFFICE/CAGROUP/run_python/task_scheduler/']

    all_test = 0
    print(f'подключение PF')
    for i in bd_pf_list:
        if os.path.exists(f'{i}'):
            df_PF = pd.read_csv(f'{i}')
            print(f'PF подключено к {i}')
            all_test += 1
            break
        else:
            print(f'подключения нет {i}')
            print(f'ищем дальше ...')
            continue

    print(f'подключение ZP')
    for i in bd_zp_list:
        if os.path.exists(f'{i}'):
            df_ZP = pd.read_csv(f'{i}')
            print(f'ZP подключено к {i}')
            all_test += 1
            break
        else:
            print(f'подключения нет {i}')
            print(f'ищем дальше ...')
            continue

    print(f'подключение шрифтов')
    for i in bd_font_dir_list:
        test = 0
        if os.path.isdir(f'{i}'):
            print(f'FPDF_шрифты подключены к {i}')
            font_dir = f'{i}'
            all_test += 1
            break
        else:
            print(f"Ошибка директории FPDF_шрифтов {i}")
            print(f'ищем дальше ...')
            continue

    if all_test == 3:
        print("БД подключены")
    else:
        print('проверьте поделючение')
    return df_PF, df_ZP, font_dir

df_PF, df_ZP, font_dir = connect_bd()
# добавим столбец с датой
df_PF['Дата'] = df_PF['год'].astype("str") + "-" + df_PF['месяц_digit'].astype("str") + "-" + '1'
df_PF['Дата'] = pd.to_datetime(df_PF['Дата'])
last_autocentre = df_PF['автоцентр'].unique()

last_mounth = df_PF[(df_PF['год'] == df_PF['год'].max()) & (df_PF['индекс'] == '210') & (df_PF['факт'] > '0')][
    'месяц_digit'].unique()

# столбцы с СУММАми в названии переводит в формат float64

def Float_columns(table):
    # находит все столбцы с Дата
    float_list = ['план', 'факт']
    float_cols = table.columns[table.columns.str.upper().str.contains('|'.join(float_list).upper())]
    # меняет тип данных на основании списка
    table[float_cols] = table[float_cols].apply(pd.to_numeric, errors='coerce')
    return table

df_PF = Float_columns(df_PF)
df_ZP['карта'] = df_ZP['Карта ав-с/зп']+df_ZP['Карта премия']
df_ZP['%_карты'] = round((df_ZP['карта']/df_ZP['Всего начислено'])*100,1)

# 2023 'ychr' + 'ym' = 'ychr'
df_PF.loc[(df_PF['год'] == 2023) & (df_PF['автоцентр'] == 'ym'), 'автоцентр'] = 'ychr'
# 2022 'ys' + 'ym' = 'ychr'
df_PF.loc[(df_PF['год'] == 2022) & (df_PF['автоцентр']=='ys'), 'автоцентр'] = 'ym'
df_PF.loc[(df_PF['год'] == 2022) & (df_PF['автоцентр']=='ym'), 'автоцентр'] = 'ychr'
# 2023 'yj' + 'yr' = 'yr'
df_PF.loc[(df_PF['год'] == 2023) & (df_PF['автоцентр']=='yj'), 'автоцентр'] = 'yr'
# 2022 'ych' + 'yr' = 'yr'
df_PF.loc[(df_PF['год'] == 2022) & (df_PF['автоцентр']=='ych'), 'автоцентр'] = 'yr'

df_PF = df_PF.groupby(['индекс', 'наименование', 'месяц','автоцентр','источник',
               'год','месяц_digit','Дата']).sum().sort_values('месяц_digit').reset_index()



# ПЕРЕМЕННЫЕ
name_ac = name_ac  # подразделение
month_pf = month_pf  # месяц int
year_pf = int(year_pf)  # год int
lim_zp_max = 100000    # лимит по ЗП

ind_pf_revenue_all = '60' # выр всего
ind_pf_revenue_AS = '70' # выр АС

ind_expenses_all = '2580н' # расходы всего
ind_expenses_AC = '2590н' # АЦ
ind_expenses_AS = '2600н' # АС
ind_expenses_TC = '2610н' # ТЦ

ind_expenses_ZP = '2630' # зарплата
ind_expenses_nalog_fot = '2630н' # налоги ФОТ
ind_expenses_rent = '2670' # аренда
ind_expenses_advertising = '2710' # реклама
ind_expenses_security = '2750' # охрана
ind_expenses_connection = '2790' # связь
ind_expenses_pzv_njd = '2830' # произв нужды
ind_expenses_hoz_njd = '2870' # хоз нужды
ind_expenses_pc_po = '2910' # комп и по
ind_expenses_proch = '2950' # прочие
ind_expenses_ub = '2990' # убытки
ind_expenses_amrt = '3030' # амортизация
ind_expenses_amrt_dop = '3080' # амортизация

# блок переменных дохода
ind_pf_doh = '210'    # доход
ind_pf_doh_as = '220' # доход АС
ind_pf_doh_tc = '260' # доход ТЦ
ind_pf_doh_ovp = 'БуДох' # доход ОВП
ind_pf_doh_prch_usl = '1420'
ind_pf_doh_otch_oup = '1514'
ind_pf_doh_meh = '1560' # доход мех
ind_pf_doh_kuz = '2180' # доход куз
ind_pf_doh_do = '2430' # доход до

ind_pf_doh_meh_rab = '270' # доход МЦ работы
ind_pf_doh_meh_zch = '280' # доход МЦ запч
ind_pf_doh_kuz_rab = '290' # дох КЦ работы
ind_pf_doh_kuz_zch = '300' # дох КЦ запч
ind_pf_doh_do_rab = '310' # дох ДО работы
ind_pf_doh_do_zch = '320' # дох ДО запч
ind_pf_doh_real_zch = '330' # дох реализация запч
ind_pf_doh_proch_ = '340' # прочее
ind_pf_doh_bonus = '350' # дох бонус ТЦ
ind_pf_doh_prch_usl_TC = '351' # прочие услуги
ind_pf_doh_bonus_program = '1532'  # дох бонус программа

# блок переменных ЗП
ind_pf_zp = '2630'    # ЗП
ind_pf_zp_AC = '2640'
ind_pf_zp_AS = '2650'
ind_pf_zp_TC = '2660'

# блок переменных ОМ
om_itog = 'ОМ'
om_auto = 'НОВДох'
om_strah = '1512'
om_ovp = '1513'
om_do = '1517'
om_privel = '1400'
om_bonus = '1510'

# блок переменных по кол-ву авто
am_count_itog = '480'
am_new_roz = 'розн_ам'
am_new_opt = 'опт_ам'
am_new_by = 'проч_ам'
am_new_prch = 'used_4'

# блок переменных ТЦ
ind_pf_nch_meh = '50' # нч мех
ind_pf_nch_kuz = '40' # нч куз
ind_pf_nch_do = '30' # нч до
ind_pf_nch_TO_R = '1800' # нч ТОиР

ind_pf_revenue_mex_itg = '110' # выручка ТЦ
ind_pf_revenue_mex_rab = '120' # выручка МЦ работы
ind_pf_revenue_mex_zch = '130' # выручка МЦ запчасти
ind_pf_revenue_kuz_rab = '140' # выручка КЦ работы
ind_pf_revenue_kuz_zch = '150' # выручка КЦ запчасти
ind_pf_revenue_do_rab = '160' # выручка ДО работы
ind_pf_revenue_do_zch = '170' # выручка ДО запчасти
ind_pf_revenue_real_zch = '180' # выручка продажа запчасти
ind_pf_revenue_proch = '190'  # выручка прочее
ind_pf_revenue_bonus_TC = '200' # бонус ТЦ
ind_pf_revenue_proch_usl = '201'  # выручка прочие услуги
ind_pf_balance_profit = '3120н' # балансовая прибыль по стар
ind_pf_balance_profit_AS = '3130н' # балансовая прибыль АС по стар
ind_pf_balance_profit_TC = '3140н' # балансовая прибыль ТЦ по стар
ind_pf_net_profit = '3290'     # ЧП
ind_pf_net_profit_AS = '3300'  # ЧП АС
ind_pf_net_profit_TC = '3310'  # ЧП ТЦ

# справочник названий в PF и ZP
spravka = {'yzs':['VCHE', 'VSUZ'], 'svr':['M'], 'cnt':['HND'], 'yzk':['VKUZ'], 'yzm':['VMUZ']}
spravka_month = {1:'Январь', 2:'Февраль', 3:'Март', 4:'Апрель',
                 5:'Май', 6:'Июнь', 7:'Июль', 8:'Август',
                 9:'Сентябрь', 10:'Октябрь',11:'Ноябрь', 12:'Декабрь'}
spravka_new_name = {'yzs': 'Чери-Сузуки', 'yzm': 'Джетур-Мазда', 'cnt':'Баик-Хендэ', 'yzk':'Киа',
                    'srt':'Омода-Джаку Саратов', 'yti':'ОВП Ярославль','yh':'Баик-Хендэ Ярославль',
                    'ychr':'Чери-Мазда-Сузуки Ярославль', 'yr':'Джак-Москва-Рено Ярославль',
                    'yv':'Фав-Джетта-Фольксваген Ярославль'}
spravka_exceptions = ('ychr','yti','yh','yr','yv','srt') # исключим в функциях факт ЗП

# КЛАССЫ

with open("process_progress.txt", "w") as file:  # заполняем прогрессбар
    file.write("10")


class Indicators:

    def __init__(self, name: str, doh_info: str, df) -> None:
        """инициализатор

        Args:
            name (str): описание индикатора (пример - Доход_итого_к_ЗП_итого)

            doh_info (str): подстрока для подстановки в столбец дохода (...._факт_дох_2023)

            df (_type_): DataFrame здесь мерджим функцииями дохода и зп
        """
        self.name = name
        self.doh_info = doh_info
        self.df = df

    def rename(self):
        """переименовывает столбец дохода, подставля переменную doh_info из __init__
        """
        self.df = self.df.rename(columns={self.df.columns[1]: f'{self.doh_info}_{self.df.columns[1]}'})
        self.df['ФЗП_от_ДОХ_%'] = round((self.df.iloc[:, 2] / self.df.iloc[:, 1]) * 100, 2)
        self.df = self.df.fillna(0)

    def info(self):
        """Информация о обьекте класса

        Returns:
            _type_: str and df (возвращает имя и сам df)
        """
        print(f"{self.name}")
        return self.df

# ФУНКЦИИ
def func_float_df_digit(df, type_: (int, float) = float):
    df = df
    """Преобразует числовые значения в int / float

    Args:
        df (_type_): df - который преобразуем
        type_ (int, float): указываем тип данных int or float

    Returns:
        _type_: DataFrame
    """

    for i in df.columns:
        try:
            df[i] = df[i].apply(lambda x: round(type_(x), 3))
            # return df
        except:
            print(f'{func_float_df_digit.__name__}: {i} не преобразован в {type_}')
    return df


def func_ZP_podr_all_year(df, podr=[], otdel=[]):
    """Функция выдает результат ЗП с начала года по подразделению и отделу

    Args:
        df (_type_ DataFrame): df по которому делаем выборку

        podr (list, optional): указываем подразделение в формате ['АЦ', 'АС', 'ТЦ'] по умолчанию выбраны все

        otdel (list, optional): указываем отдел в формате ['АЦ.Рук', 'АС.Рук', 'ТЦ.рук', 'ФО', 'АЦ.общ', 'ОПА', 'АС.общ',
     'МЦ', 'ИТР.МЦ', 'ДО', 'ОЗЧ', 'КЦ', 'ИТР.КЦ', 'ТЦ.общ', 'ОВП'] по умолчанию выбраны все


    Returns:
        _type_: DataFrame
    """
    df_ZP = df
    time_t = df_ZP[(df_ZP['Год'] == year_pf) & (df_ZP['Месяц'] <= month_pf)]
    podr = podr if len(podr) > 0 else time_t['Подразделение'].unique()
    otdel = otdel if len(otdel) > 0 else time_t['Отдел'].unique()

    time_table = df_ZP[(df_ZP['Год'] == year_pf) &
                       (df_ZP['Месяц'] <= month_pf) &
                       (df_ZP['Подразделение'].isin(podr)) &
                       (df_ZP['Отдел'].isin(otdel)) &
                       (df_ZP['Автоцентр'].isin(spravka.get(name_ac)))][['Месяц', 'Всего начислено']].groupby(
        'Месяц').agg({'Всего начислено': 'sum'}).reset_index()
    time_table['Месяц'] = time_table['Месяц'].apply(lambda x: spravka_month.get(x))
    time_table = time_table.rename(columns={'Месяц': f'месяц'})
    # строка ниже возвращает в шапке название подразделений и отделов
    # time_table = time_table.rename(columns={'Всего начислено': f'Всего_нач_подр_{podr}_отделы_{otdel}'})

    return time_table


def func_ZP_podr_month(df, podr=[], otdel=[]):
    """Функция выдает результат ЗП за текущий месяц по подразделению и отделу

    Args:
        df (_type_ DataFrame): df по которому делаем выборку

        podr (list, optional): указываем подразделение в формате ['АЦ', 'АС', 'ТЦ'] по умолчанию выбраны все

        otdel (list, optional): указываем отдел в формате ['АЦ.Рук', 'АС.Рук', 'ТЦ.рук', 'ФО', 'АЦ.общ', 'ОПА', 'АС.общ',
     'МЦ', 'ИТР.МЦ', 'ДО', 'ОЗЧ', 'КЦ', 'ИТР.КЦ', 'ТЦ.общ', 'ОВП'] по умолчанию выбраны все


    Returns:
        _type_: float
    """
    df_ZP = df
    time_t = df_ZP[(df_ZP['Год'] == year_pf) & (df_ZP['Месяц'] <= month_pf)]
    podr = podr if len(podr) > 0 else time_t['Подразделение'].unique()
    otdel = otdel if len(otdel) > 0 else time_t['Отдел'].unique()

    time_table = df_ZP[(df_ZP['Год'] == year_pf) &
                       (df_ZP['Месяц'] == month_pf) &
                       (df_ZP['Подразделение'].isin(podr)) &
                       (df_ZP['Отдел'].isin(otdel)) &
                       (df_ZP['Автоцентр'].isin(spravka.get(name_ac)))][['Месяц', 'Всего начислено']].groupby(
        'Месяц').agg({'Всего начислено': 'sum'}).reset_index()
    time_table['Месяц'] = time_table['Месяц'].apply(lambda x: spravka_month.get(x))
    time_table = time_table.rename(columns={'Месяц': f'месяц'})
    # строка ниже возвращает в шапке название подразделений и отделов
    # time_table = time_table.rename(columns={'Всего начислено': f'Всего_нач_подр_{podr}_отделы_{otdel}'})
    try:
        x = round(float(time_table.iloc[0, 1]), 1)
        return x
    except:
        return 0


def func_IND_podr(df, ind: str = None, fkt_or_pln='факт'):
    """Возвращает данные по индексу из PF за текущий месяц

    Args:
        df (_type_): df

        ind (str, optional): подаем индекс тип 'str'

        fkt_or_pln = 'факт' или 'план' по умолчанию факт

    Returns:
        _type_: float or 0
    """
    fkt_or_pln = fkt_or_pln.lower()
    df_PF = df
    try:
        x = df_PF[(df_PF['индекс'].isin([ind])) &
                  (df_PF['автоцентр'] == name_ac) &
                  (df_PF['год'] == year_pf) &
                  (df_PF['месяц_digit'] == month_pf)][[fkt_or_pln]].reset_index().iloc[0, 1]

        return round(float(x), 1)
    except:
        return 0


def func_IND_podr_year(df, ind='210', fkt_or_pln='факт', ren='дох', year: int = year_pf):
    """Возвращает данные по индексу из PF с начала года

    Args:
        df (_type_): df

        ind (str, optional): индекс по умолчанию '210'.

        fkt_or_pln (str, optional): прописать 'план' или 'факт'.

        ren (str): указываем префикс названия столбца факт_<..>_2023

        year (int): год по умолчанию текущий

    Returns:
        _type_: df
    """
    df_PF = df
    try:
        result = df_PF[(df_PF['индекс'] == ind) &
                       (df_PF['автоцентр'] == name_ac) &
                       (df_PF['год'] == year) &
                       (df_PF['месяц_digit'] <= month_pf)][['месяц', fkt_or_pln]].fillna(0).drop_duplicates()
        result = result.rename(columns={f'{fkt_or_pln}': f'{fkt_or_pln}_{ren}_{year}'})
        result = func_float_df_digit(result, float)
        return result
    except:
        return print(f'функция {func_IND_podr_year.__name__} вернулась с ошибкой')


def func_limit_ZP(df, limit: int = lim_zp_max, podr=[]):
    """Возвращает ЗП подразделения за месяц превышающую лимит

    Args:
        df (_type_): df

        limit (int, optional): устанавливваем лимит, по умолчанию 85000

        podr: lst

    Returns:
        _type_: df по умолчанию все но можно выбирать ['АЦ','АС','ТЦ']
    """

    df_ZP = df
    time_t = df_ZP[(df_ZP['Год'] == year_pf) & (df_ZP['Месяц'] <= month_pf)]
    podr = podr if len(podr) > 0 else time_t['Подразделение'].unique()
    # столбцы которые включаем
    fix_ = ['ФИО', 'Должность', 'Оклад', 'Рачетная ЗП', 'Вычет-Премия', 'Отпуск',
            'Всего начислено', 'карта', '%_карты', 'Подразделение', 'Отдел']
    try:
        result = df_ZP[(df_ZP['Год'] == year_pf) &
                       (df_ZP['Месяц'] == month_pf) &
                       (df_ZP['Подразделение'].isin(podr)) &
                       (df_ZP['Автоцентр'].isin(spravka.get(name_ac))) &
                       (df_ZP['Всего начислено'] >= limit)][[i for i in df_ZP.columns if i in fix_]]
        result = result.rename(columns={'ФИО': f'ФИО ЗП >= {limit}'})
        result = result[[f'ФИО ЗП >= {limit}', 'Должность', 'Оклад', 'Рачетная ЗП', 'Вычет-Премия', 'Отпуск',
                         'Всего начислено', 'карта', '%_карты', 'Подразделение', 'Отдел']]
        return result.fillna(0)
    except:
        print(f'функция {func_limit_ZP.__name__} вернулась с ошибкой')


def func_ZP_count_pers(df, month: int, podr=[], otdel=[], flg=None):
    """Количество Сотрудников

    Args:
        df (_type_): df
        month (int): месяц int - обратить внимиание на flg для отображения за один месяц проставить 1 по умолчанию выдает с начала года

        podr (list, optional): выбрать АЦ ТЦ АС по умолчанию если пусто проставляет все подразеделения [] проставляет все подразеделения.

        otdel (list, optional): выбрать КЦ, ИТР.КЦ, МЦ, ДО и тд по умолчанию если пусто [] выбирает все подразделения.

        flg (_type_, optional): по умолчанию None, значит все подразделения с начала года, для месяца выбрать 1.

    Returns:
        _type_: если flg =1 - float / flg =None lst(int, int ....)
    """

    df_ZP = df
    tt_time = df_ZP[(df_ZP['Год'] == year_pf) & (df_ZP['Месяц'] <= month_pf)]['Месяц'].unique()

    if flg == 1:
        df_ZP = df_ZP.fillna('0')
        podr = podr if len(podr) > 0 else df_ZP['Подразделение'].unique()
        otdel = otdel if len(otdel) > 0 else df_ZP['Отдел'].unique()
        df_per = df_ZP[(df_ZP['Год'] == year_pf) & (df_ZP['Месяц'] == month) & (
            df_ZP['Автоцентр'].isin(spravka.get(name_ac)))]  # сортирем по году месяцу и АЦ
        df_per = df_per[~df_per['Должность'].str.contains('доплат|бонус')]  # исключаем доплаты и бонусы в должностях
        # отбираем индексы основных сотрудников сотрудников, исключая повторения с меньшей ЗП - то есть совмещений
        osn_person = list(
            df_per.sort_values(by=['ФИО', 'Всего начислено'], ascending=False).drop_duplicates(['ФИО'], keep='first')[
                ['Таб №']].iloc[:, 0])
        # добавили столбец - применили функцию
        df_per['основной'] = df_per['Таб №'].apply(lambda x: 1 if int(x) in osn_person else 0)

        return df_per[(df_per['Подразделение'].isin(podr)) & (df_per['Отдел'].isin(otdel))]['основной'].sum()

    else:
        res = []
        for i in tt_time:
            df_ZP = df_ZP.fillna('0')
            podr = podr if len(podr) > 0 else df_ZP['Подразделение'].unique()
            otdel = otdel if len(otdel) > 0 else df_ZP['Отдел'].unique()
            df_per = df_ZP[(df_ZP['Год'] == year_pf) & (df_ZP['Месяц'] == i) & (
                df_ZP['Автоцентр'].isin(spravka.get(name_ac)))]  # сортирем по году месяцу и АЦ
            df_per = df_per[
                ~df_per['Должность'].str.contains('доплат|бонус')]  # исключаем доплаты и бонусы в должностях
            # отбираем индексы основных сотрудников сотрудников, исключая повторения с меньшей ЗП - то есть совмещений
            osn_person = list(
                df_per.sort_values(by=['ФИО', 'Всего начислено'], ascending=False).drop_duplicates(['ФИО'],
                                                                                                   keep='first')[
                    ['Таб №']].iloc[:, 0])
            # добавили столбец - применили функцию
            df_per['основной'] = df_per['Таб №'].apply(lambda x: 1 if int(x) in osn_person else 0)

            res.append(df_per[(df_per['Подразделение'].isin(podr)) & (df_per['Отдел'].isin(otdel))]['основной'].sum())
        return res


def convert_str(df, resurs=1):
    """Конвертирует df в tuple str для дальнейшего применения в FPDF

    Args:
        df (_type_): df

        resurs (int, optional): 1 or 2 (1 это шапка и df в разных кортежах, 2 шапка и df в одном кортеже)

    Returns:
        _type_: tuple - (в зависимости от resurs подаставлять 1 или 2 переменных)
    """
    df = df
    if resurs == 1:
        lsty = []
        tab_col = tuple(str(i) for i in df.columns)
        for i in range(df.shape[0]):
            lsty.append(tuple(str(i) for i in df.iloc[i, :]))
        tab_col = tuple(tab_col)
        lsty = tuple(lsty)
        return tab_col, lsty

    elif resurs == 2:
        lsty = []
        lsty.append(tuple(str(i) for i in df.columns))
        for i in range(df.shape[0]):
            lsty.append(tuple(str(i) for i in df.iloc[i, :]))

        lsty = tuple(lsty)
        return lsty


def result_plan(x: int):
    '''
    функция для проверки выполнения плана, принимает один параметр
    '''
    try:
        return f'{x} % ↗' if x >= 100 else f'{x} % ↘'
    except:
        return type(i)

def infinity_(df):
    """ Убирает значение/ошибки inf из df

    Args:
        df (_type_): df

    Returns:
        _type_: df без inf
    """
    df = df
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def convertor_zeros(df, combine=1000000, rnd=3, column=[]):
    """ Функция преобразует в млн и прочие ед, округляет значения
    есть возможность выбирать столбцы для конвертиации
    по умолчанию ковертирует все цифры в df

    Args:
        df (_type_): df

        combine (int, optional): число на которое делимм Defaults to 1000000.

        rnd (int, optional): округление. Defaults to 1000000.

        column (list, optional): столбцы которые конвертирем по умолчанию все
        . Defaults to [].

    Returns:
        _type_: df
    """
    df = df
    column = df.columns if len(column) == 0 else column
    for i in column:
        if '%' not in i:
            try:
                df[i] = df[i].apply(lambda x: round(Decimal(x) / combine, rnd))
            except:
                print(f'{i} не удалось преобразовать')
    return df

def mln_x(x):
    return round(float(x)/1000000, 1)

def dynamic_2_years(df, ind):
    """Поучаем показатели с начала прошлого года по месяц текущего

    Args:
        df (_type_): df
        ind (_type_): индекс показателя

    Returns:
        _type_: _description_
    """
    res = df[(df['индекс'] == ind )
             & (df['автоцентр'] == name_ac)
             & (df['год'] >= year_pf-1)
             & (df['Дата'] <= f'{year_pf}-{month_pf}-01')][['Дата','факт','план']].fillna(0)
    res['факт'] = res['факт'].apply(mln_x)
    res['план'] = res['план'].apply(mln_x)
    res = res.sort_values('Дата')
    return res


# плановый доход
doh_plan = func_IND_podr(df_PF, ind_pf_doh, 'план')
# фактический доход
doh_fact = func_IND_podr(df_PF, ind_pf_doh, 'факт')
# плановая ЗП
zp_plan = func_IND_podr(df_PF, ind_pf_zp, 'план')
# фактическая ЗП
zp_fact = func_IND_podr(df_PF, ind_pf_zp, 'факт')


row1 = pd.Series([doh_plan, zp_plan, doh_fact, zp_fact])
df_table_1 = pd.DataFrame([row1])
df_table_1.columns = ['Доход_план', 'ЗП_план', 'Доход_факт', 'ЗП_факт']
df_table_1['Доля_ЗП_от_дохода_план'] = round((df_table_1['ЗП_план'] / df_table_1['Доход_план'])*100, 2)
df_table_1['Доля_ЗП_от_дохода_факт'] = round((df_table_1['ЗП_факт'] / df_table_1['Доход_факт'])*100, 2)
df_table_1['+/-'] = round(df_table_1['Доля_ЗП_от_дохода_факт'] - df_table_1['Доля_ЗП_от_дохода_план'], 2)
df_table_1['+/-'] = df_table_1['+/-'].apply(lambda x: f"{x} % превышение" if x > 0 else f"{x} % первышения нет")
df_table_1['Доля_ЗП_от_дохода_план'] = df_table_1['Доля_ЗП_от_дохода_план'].apply(lambda x: f"{x} %")
df_table_1['Доля_ЗП_от_дохода_факт'] = df_table_1['Доля_ЗП_от_дохода_факт'].apply(lambda x: f"{x} %")

# меняем разрядность только у исходных столбцов
for i in ['Доход_план', 'ЗП_план', 'Доход_факт', 'ЗП_факт']:
    try:
        df_table_1[i] = df_table_1[i].apply(lambda x: '{0:,}'.format(int(x)).replace(',', ' '))
    except:
        print(f'Ошибка изменения разрядности по столбцу {i}')

# dfi.export(df_table_1, 'df_распоряжение_оптимизации.png', dpi=100)

with open("process_progress.txt", "w") as file:  # заполняем прогрессбар
    file.write("20")


with plt.style.context('seaborn-v0_8'):  # применим стиль bmh / fivethirtyeight / seaborn-deep

    month_hisogr_ = [i for i in df_PF[df_PF['месяц_digit'] <= month_pf][
        'месяц'].unique()]  # фильтруем df по месяцу отчета выбираем месяца
    doh_autocentr_all = [round(float(
        df_PF[(df_PF['индекс'] == ind_pf_doh) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
            ['факт', 'месяц']]
        .fillna(0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]
    MMM = [f"{i} \n всего \n {round(y, 1)}" for i, y in zip(month_hisogr_, doh_autocentr_all)]
    month_hisogr_ = MMM

    doh_autocentr = {
        'Доход ТЦ': np.array([round(float(df_PF[(df_PF['индекс'] == ind_pf_doh_tc) & (df_PF['автоцентр'] == name_ac)
                                                & (df_PF['год'] == year_pf)][['факт', 'месяц']].drop_duplicates(
            'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]),
        'Доход АС': np.array([round(float(df_PF[(df_PF['индекс'] == ind_pf_doh_as) & (df_PF['автоцентр'] == name_ac)
                                                & (df_PF['год'] == year_pf)][['факт', 'месяц']].drop_duplicates(
            'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)])

    }
    width = 0.6  # толщина свеч

    fig, ax = plt.subplots(figsize=(0.5 + (month_pf * 1.2), 7), dpi=200)  # ширина по кол-ву месяцев month_pf
    ax.tick_params(labelsize=14)  # шрифт осей
    plt.rcParams['font.size'] = '16'  # шрифт в данных
    bottom = np.zeros(month_pf)  # по какому числу месяцев строим # month_pf
    for as_tc, doh_sum in doh_autocentr.items():
        p = ax.bar(month_hisogr_, doh_sum, width, label=as_tc, bottom=bottom)
        bottom += doh_sum
        ax.bar_label(p, label_type='center')

    ax.set_title(f'Динамика дохода \n по АЦ {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf} в млн.руб',
                 fontweight="bold", fontsize=10)
    ax.grid(True, linestyle=':')  # линии разметки
    ax.legend()
    plt.legend()
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig('Динамика_дохода_1.png')

with plt.style.context('seaborn-v0_8'):  # применим стиль
    # факт предыдущего года
    left_last_fact = pd.DataFrame(
        {'месяц': [i for i in df_PF[df_PF['месяц_digit'] <= month_pf]['месяц'].unique()],
         f"{year_pf - 1}_факт": [round(float(df_PF[(df_PF['индекс'] == ind_pf_doh) & (df_PF['автоцентр'] == name_ac) &
                                                   (df_PF['год'] == year_pf - 1)][['факт', 'месяц']].drop_duplicates(
             'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]})

    # факт текущего года
    right_first_fact = pd.DataFrame({'месяц': [i for i in df_PF[df_PF['месяц_digit'] <= month_pf]['месяц'].unique()],
                                     f"{year_pf}_факт": [round(
                                         float(df_PF[(df_PF['индекс'] == ind_pf_doh) & (df_PF['автоцентр'] == name_ac) &
                                                     (df_PF['год'] == year_pf)][['факт', 'месяц']].drop_duplicates(
                                             'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]})
    # план текущего года
    right_first_pact = pd.DataFrame(
        {'месяц': [i for i in df_PF[df_PF['месяц_digit'] <= month_pf]['месяц'].unique()],
         f"{year_pf}_план": [round(float(df_PF[(df_PF['индекс'] == ind_pf_doh) &
                                               (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
                                             ['план', 'месяц']].drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i
                             in range(month_pf)]})

    # джойним предыдущи и текущий + план текущего
    df_last_first = pd.merge(left_last_fact, right_first_fact, how='outer')
    df_last_first_target = pd.merge(df_last_first, right_first_pact, how='outer')

    # строим график
    ax = df_last_first_target[['месяц', f"{year_pf}_план"]].plot(
        x='месяц', linestyle='-', marker='o', figsize=(0.5 + (month_pf) * 1.2, 7), color='black', fontsize=9)
    df_last_first_target[['месяц', f"{year_pf - 1}_факт", f"{year_pf}_факт"]].plot(x='месяц', kind='bar',
                                                                                   ax=ax, fontsize=9)
    ax.tick_params(labelsize=14)  # шрифт осей
    plt.rcParams['font.size'] = '16'  # шрифт в данных
    ax.set_title(
        f'Динамика дохода \n по {spravka_new_name.get(name_ac, name_ac)} всего \n за {year_pf - 1} - {year_pf}г.',
        fontweight="bold", fontsize=10)
    ax.grid(True, linestyle=":")  # линии разметки
    plt.ylabel('млн.руб', fontsize=10, fontweight="bold")
    plt.xlabel('месяц', fontsize=10, fontweight="bold")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig('Динамика_дохода_2.png')  # сохряняет в картику

f_lst_f_tg_1 = pd.merge(func_IND_podr_year(df_PF, ind_pf_doh, 'факт', year=year_pf - 1),
                        func_IND_podr_year(df_PF, ind_pf_doh, 'план', year=year_pf))
f_lst_f_tg_1 = pd.merge(f_lst_f_tg_1, func_IND_podr_year(df_PF, ind_pf_doh, 'факт', year=year_pf))
f_lst_f_tg_1.loc['ИТОГО'] = f_lst_f_tg_1[
    [f"факт_дох_{year_pf - 1}", f"факт_дох_{year_pf}", f"план_дох_{year_pf}"]].sum()
f_lst_f_tg_1['Ф_к_П_%'] = round((f_lst_f_tg_1[f"факт_дох_{year_pf}"] / f_lst_f_tg_1[f"план_дох_{year_pf}"]) * 100, 1)
f_lst_f_tg_1[f'Ф_{year_pf}_к_{year_pf - 1}_%'] = round(
    (f_lst_f_tg_1[f"факт_дох_{year_pf}"] / f_lst_f_tg_1[f"факт_дох_{year_pf - 1}"]) * 100, 1)
f_lst_f_tg_1[f'ДОХ_с_нак_Ф_{year_pf - 1}'] = round(f_lst_f_tg_1[f"факт_дох_{year_pf - 1}"].cumsum())
f_lst_f_tg_1[f'ДОХ_с_нак_Ф_{year_pf}'] = round(f_lst_f_tg_1[f"факт_дох_{year_pf}"].cumsum())
f_lst_f_tg_1[f'нак_{year_pf}_к_{year_pf - 1}_%'] = round(
    (f_lst_f_tg_1[f'ДОХ_с_нак_Ф_{year_pf}'] / f_lst_f_tg_1[f'ДОХ_с_нак_Ф_{year_pf - 1}']) * 100, 1)

for i in [i for i in f_lst_f_tg_1.columns if '%' in i]:
    f_lst_f_tg_1[i] = f_lst_f_tg_1[i].apply(result_plan)

f_lst_f_tg_1 = convertor_zeros(f_lst_f_tg_1)

# удаляем лишнее по последней строке
f_lst_f_tg_1.loc['ИТОГО', f'ДОХ_с_нак_Ф_{year_pf - 1}'] = '---'
f_lst_f_tg_1.loc['ИТОГО', f'ДОХ_с_нак_Ф_{year_pf}'] = '---'
f_lst_f_tg_1.loc['ИТОГО', f'нак_{year_pf}_к_{year_pf - 1}_%'] = '---'

f_lst_f_tg_1 = f_lst_f_tg_1.fillna(0)

df_last_first_target_x = f_lst_f_tg_1.copy()
dfi.export(df_last_first_target_x, 'df_дох_с_накопл.png', dpi=200)

with plt.style.context('seaborn-v0_8'): # применим стиль bmh / fivethirtyeight / seaborn-deep
    month_hisogr = [i for i in df_PF[df_PF['месяц_digit']<= month_pf]['месяц'].unique()]  # фильтруем df по месяцу отчета выбираем месяца
    doh_autocentr = {
        'ОМ_ИТОГО': np.array(
            [round(float(df_PF[(df_PF['индекс'] == om_itog)
                                                & (df_PF['автоцентр'] == name_ac)
                                                & (df_PF['год'] == year_pf)][[
                                                    'факт', 'месяц']].fillna(0).drop_duplicates('месяц').iloc[i,0])/1000000, 2) for i in range(month_pf)]),

    }
    width = 0.6  # толщина свеч

    fig, ax = plt.subplots(figsize=(0.5+(month_pf*1.2),7), dpi=200) # ширина по кол-ву месяцев month_pf
    bottom = np.zeros(month_pf) # по какому числу месяцев строим # month_pf
    ax.tick_params(labelsize = 14) # шрифт осей
    plt.rcParams['font.size'] = '16' # шрифт в данных
    for as_tc, doh_sum in doh_autocentr.items():
        p = ax.bar(month_hisogr, doh_sum, width, label=as_tc, bottom=bottom)
        bottom += doh_sum

        ax.bar_label(p, label_type='center')
    # добавляем ось X
    plt.axhline(y=0, color='red', linestyle='--')
    ax.set_title(f'ОМ \n {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf} в млн.руб', fontweight="bold", fontsize = 9)
    ax.grid(True, linestyle=':')  # линии разметки
    ax.legend()

    plt.tight_layout()
    plt.savefig('ОМ_1.png') # сохряняет в картинку

om_result_dinamic = dynamic_2_years(df_PF, om_itog)

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 6), dpi=200)  # размер графика
    plt.plot(om_result_dinamic['Дата'], om_result_dinamic['факт'], label='ОМ_факт', color='green')
    # plt.plot(om_result_dinamic['Дата'], om_result_dinamic['план'], label='ОМ_план', color='red') # эту строку убрать

    # градус подписей по оси Х
    plt.xticks(rotation=0)
    # добавляем ось X
    plt.axhline(y=0, color='red', linestyle='--')

    plt.legend()
    plt.ylabel('Доход', fontsize=10, fontweight="bold")
    plt.xlabel('ПЕРИОД', fontsize=10, fontweight="bold")
    plt.title(
        f"Динамика операционной маржи(ОМ)  \n по {spravka_new_name.get(name_ac, name_ac)} за {year_pf - 1}-{year_pf}г в млн.руб",
        fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig('ОМ_2.png')  # сохряняет в картику

with plt.style.context('seaborn-v0_8'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    # уникальные месяца
    mmm = [i for i in df_PF[df_PF['месяц_digit'] <= month_pf]['месяц'].unique()]
    # кол-во авто всего
    itogo_autom = [round(float(
        df_PF[(df_PF['индекс'] == am_count_itog) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
            ['факт', 'месяц']].drop_duplicates('месяц').iloc[i, 0]) / 1, 2) for i in range(month_pf)]
    # собираем текст всего кол-во + мес с переносом
    MMM = [f"{i} \n всего \n {round(y)}" for i, y in zip(mmm, itogo_autom)]

    month_hisogr = MMM  # фильтруем df по месяцу отчета выбираем месяца
    doh_autocentr = {
        'АМ_розница': np.array([round(float(
            df_PF[(df_PF['индекс'] == am_new_roz) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
                ['факт', 'месяц']].drop_duplicates('месяц').iloc[i, 0]) / 1, 2) for i in range(month_pf)]),
        'АМ_опт': np.array([round(float(
            df_PF[(df_PF['индекс'] == am_new_opt) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
                ['факт', 'месяц']].drop_duplicates('месяц').iloc[i, 0]) / 1, 2) for i in range(month_pf)]),
        'АМ_АСП_демо': np.array([round(float(
            df_PF[(df_PF['индекс'] == am_new_by) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
                ['факт', 'месяц']].drop_duplicates('месяц').iloc[i, 0]) / 1, 2) for i in range(month_pf)])

    }

    doh_autocentr_it = {
        'ИТОГО': np.array([round(float(
            df_PF[(df_PF['индекс'] == am_count_itog) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
                ['факт', 'месяц']].drop_duplicates('месяц').iloc[i, 0]) / 1, 2) for i in range(month_pf)])}

    width = 0.6  # толщина свеч

    fig, ax = plt.subplots(figsize=(0.5 + (month_pf * 1.2), 7), dpi=300)  # ширина по кол-ву месяцев month_pf
    bottom = np.zeros(month_pf)  # по какому числу месяцев строим # month_pf
    ax.tick_params(labelsize=14)  # шрифт осей
    plt.rcParams['font.size'] = '16'  # шрифт в данных
    for as_tc, doh_sum in doh_autocentr.items():
        p = ax.bar(month_hisogr, doh_sum, width, label=as_tc, bottom=bottom)
        bottom += doh_sum
        ax.bar_label(p, label_type='center')

    ax.set_title(f'Распределение авто по \n {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf} г. в шт.',
                 fontweight="bold", fontsize=10)
    ax.grid(True, linestyle=':')  # линии разметки
    ax.legend()
    plt.tight_layout()
    plt.savefig('Распределение_ам.png')  # сохряняет в картику

with plt.style.context('seaborn-v0_8'):  # применим стиль bmh / fivethirtyeight / seaborn-deep /seaborn-v0_8
    mmm = [i for i in df_PF[df_PF['месяц_digit'] <= month_pf]['месяц'].unique()]
    itogo_autom = [round(float(
        df_PF[(df_PF['индекс'] == om_itog) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
            ['факт', 'месяц']].fillna(0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]
    MMM = [f"{i[0:3]}\n всего\n {round(y, 2)}" for i, y in zip(mmm, itogo_autom)]

    month_doh = MMM  # фильтруем df по месяцу отчета выбираем месяца
    doh_autocentr = {
        'ОМ_авто': np.array([round(float(
            df_PF[(df_PF['индекс'] == om_auto) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
                ['факт', 'месяц']].fillna(0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in
                             range(month_pf)]),
        'ОМ_до': np.array([round(float(
            df_PF[(df_PF['индекс'] == om_do) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
                ['факт', 'месяц']].fillna(0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in
                           range(month_pf)]),
        'ОМ_страх': np.array([round(float(
            df_PF[(df_PF['индекс'] == om_strah) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
                ['факт', 'месяц']].fillna(0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in
                              range(month_pf)]),
        'ОМ_овп': np.array([round(float(
            df_PF[(df_PF['индекс'] == om_ovp) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
                ['факт', 'месяц']].fillna(0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in
                            range(month_pf)]),
        'ОМ_бонус': np.array([round(float(
            df_PF[(df_PF['индекс'] == om_bonus) & (df_PF['автоцентр'] == name_ac) & (df_PF['год'] == year_pf)][
                ['факт', 'месяц']].fillna(0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in
                              range(month_pf)])
    }

    x = np.arange(len(month_doh))  # the label month
    width = 0.15  # ширина полос
    multiplier = 0

    fig, ax = plt.subplots(figsize=(0.5 + (month_pf * 1.2), 10), dpi=200)

    for name_doh, doh_value in doh_autocentr.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, doh_value, width, label=name_doh)
        ax.bar_label(rects, rotation=90, padding=3, fontsize=16)
        multiplier += 1.3  # растояние между столбцами одной группы

    # скрипт определяющий построение графика по оси y min max для set_ylim
    mx = 0
    mi = 0
    for k, v in doh_autocentr.items():
        if max(v) > mx:
            mx = max(v)
        if min(v) < mi:
            mi = min(v)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('млн.руб', fontsize=16, fontweight="bold")
    ax.set_title(f'Распред-ие ОМ \n по PF в млн.р', fontweight="bold", fontsize=13)
    ax.set_xticks(x + width, month_doh)
    ax.legend(loc='upper left', ncols=1)  # ncols=1 - столбцы легенды
    ax.set_ylim(mi - 3, mx + 3)  # + корреткировка +2 млн

    plt.xticks(rotation=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('месяц / доход', fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig('ОМ_3.png')  # сохряняет в картику


with open("process_progress.txt", "w") as file:  # заполняем прогрессбар
    file.write("30")


# собираем первый фрейм с ОМ
OM_rspred = df_PF[(df_PF['индекс'] == om_itog) & (df_PF['автоцентр'] == name_ac)
                  & (df_PF['год'] == year_pf)][['месяц', 'факт']].fillna(0).drop_duplicates('месяц')
OM_rspred = OM_rspred.rename(columns={'факт':f'ОМ_{year_pf}'})
#OM_rspred[f'ОМ_{year_pf}'] = OM_rspred[f'ОМ_{year_pf}'].apply(lambda x: round(Decimal(x)/1000000,3))

# к первому фрейму мерджим составные бонусные части
for i in ['НОВДох', '1512', '1513', '1517', '1400', '1510']:
    OM_rspred = pd.merge(OM_rspred, df_PF[(df_PF['индекс'] == i)
                                          & (df_PF['автоцентр'] == name_ac)
                                          & (df_PF['год'] == year_pf)][['месяц', 'факт']].fillna(0).drop_duplicates('месяц'))
    OM_rspred = OM_rspred.rename(columns={'факт':f'{i}'})
    OM_rspred[f'{i}'] = OM_rspred[f'{i}'].apply(lambda x: round(Decimal(x)/1000000,3))

# переименовываем
OM_rspred = OM_rspred.rename(columns={'НОВДох':'ОМ_авто', '1512':'ОМ_страх', '1513':'ОМ_овп', '1517':'ОМ_до', '1400':'ОМ_привелегии', '1510':'ОМ_бонус'})

# блок сборки новых авто розница + опт
df_count_rozn =  df_PF[(df_PF['индекс'] == am_new_roz)
                       & (df_PF['автоцентр'] == name_ac)
                       & (df_PF['год'] == year_pf)][['месяц', 'факт']].fillna(0).drop_duplicates('месяц')
df_count_rozn = df_count_rozn.rename(columns={'факт':'ам_розн'})
df_count_opt =  df_PF[(df_PF['индекс'] == am_new_opt)
                      & (df_PF['автоцентр'] == name_ac)
                      & (df_PF['год'] == year_pf)][['месяц', 'факт']].fillna(0).drop_duplicates('месяц')
df_count_opt = df_count_opt.rename(columns={'факт':'ам_опт'})
df_auto_new = pd.merge(df_count_rozn, df_count_opt, how='left')
df_auto_new['ам_новые(розн_опт)'] = df_auto_new['ам_розн'].apply(lambda x: round(float(x),1)) + df_auto_new['ам_опт'].apply(lambda x: round(float(x),3))
df_auto_new = pd.DataFrame(df_auto_new[['месяц','ам_новые(розн_опт)']])

# обединяем два фрейма
OM_rspred = pd.merge(OM_rspred, df_auto_new, how='left')
# обрезаем табл по строкам без nan и по всем столбцам
OM_rspred['кум_на_1_ам'] = round(OM_rspred[f'ОМ_{year_pf}'] / OM_rspred['ам_новые(розн_опт)'], 2)
OM_rspred.replace([np.inf, -np.inf], np.nan, inplace=True) # убираем ошибки inf
OM_rspred = OM_rspred.iloc[0:month_pf,0:len(OM_rspred.columns)]
OM_rspred[f'ОМ_{year_pf}'] = OM_rspred[f'ОМ_{year_pf}'].apply(lambda x: round(Decimal(x)/1000000,3))
OM_rspred[f'кум_на_1_ам'] = OM_rspred[f'кум_на_1_ам'].apply(lambda x: round(Decimal(x)/1000000,3))
OM_rspred[f'ам_новые(розн_опт)'] = OM_rspred[f'ам_новые(розн_опт)'].apply(lambda x: round(Decimal(x)))
dfi.export(OM_rspred, 'df_Распределение_доходности_ОМ.png', dpi=150)

OM_rspred_tabl = OM_rspred.copy()

# блок сборки кумсум
OM_rspred[f'ОМ_накопительно_{year_pf}'] = OM_rspred[f'ОМ_{year_pf}'].astype(float).cumsum()
OM_rspred[f'АМ_накопительно_{year_pf}'] = OM_rspred[f'ам_новые(розн_опт)'].astype(float).cumsum()
OM_rspred_ = OM_rspred.copy()
OM_rspred_ = OM_rspred_[['месяц',f'ОМ_накопительно_{year_pf}', f'АМ_накопительно_{year_pf}']]
OM_rspred_['кум_на_1_ам'] = round(OM_rspred_[f'ОМ_накопительно_{year_pf}'] / OM_rspred_[f'АМ_накопительно_{year_pf}'], 2)
OM_rspred_ = infinity_(OM_rspred_)
OM_rspred_[f'АМ_накопительно_{year_pf}'] = OM_rspred_[f'АМ_накопительно_{year_pf}'].apply(lambda x: round(x))
OM_rspred_[f'ОМ_накопительно_{year_pf}'] = OM_rspred_[f'ОМ_накопительно_{year_pf}'].apply(lambda x: round(Decimal(x), 3))
OM_rspred_.fillna(0)
OM_rspred_['кум_на_1_ам'] = OM_rspred_['кум_на_1_ам'].apply(lambda x: round(Decimal(x), 3))
OM_rspred_nacopl = OM_rspred_.copy()
#OM_rspred_


result_doh_AC = df_PF[(df_PF['индекс'] == ind_pf_doh )  & (df_PF['автоцентр'] == name_ac) &
                          (df_PF['год'] == year_pf) & (df_PF['месяц_digit'] <= month_pf)][['месяц','план','факт']].fillna(0)
result_doh_AC = result_doh_AC.rename(columns={'факт':f'факт_дох_АЦ_{year_pf}', 'план':f'план_дох_АЦ_{year_pf}'})
result_doh_AS = df_PF[(df_PF['индекс'] == ind_pf_doh_as )  & (df_PF['автоцентр'] == name_ac) &
                          (df_PF['год'] == year_pf) & (df_PF['месяц_digit'] <= month_pf)][['месяц','план','факт']].fillna(0)
result_doh_AS = result_doh_AS.rename(columns={'факт':f'факт_дох_АС_{year_pf}', 'план':f'план_дох_АС_{year_pf}'})
result_doh_TC = df_PF[(df_PF['индекс'] == ind_pf_doh_tc )  & (df_PF['автоцентр'] == name_ac) &
                          (df_PF['год'] == year_pf) & (df_PF['месяц_digit'] <= month_pf)][['месяц','план','факт']].fillna(0)
result_doh_TC = result_doh_TC.rename(columns={'факт':f'факт_дох_ТЦ_{year_pf}', 'план':f'план_дох_ТЦ_{year_pf}'})
result_doh_all = pd.merge(result_doh_AC, result_doh_AS, how='left')
result_doh_all = pd.merge(result_doh_all, result_doh_TC, how='left')
result_ZP_itog = df_PF[(df_PF['индекс'] == ind_pf_zp )  & (df_PF['автоцентр'] == name_ac) &
                          (df_PF['год'] == year_pf) & (df_PF['месяц_digit'] <= month_pf)][['месяц','план','факт']].fillna(0)
result_ZP_itog = result_ZP_itog.rename(columns={'факт':f'факт_ЗП_ИТОГО_{year_pf}', 'план':f'план_ЗП_ИТОГО_{year_pf}'})
result_ZP_AC = df_PF[(df_PF['индекс'] == ind_pf_zp_AC )  & (df_PF['автоцентр'] == name_ac) &
                          (df_PF['год'] == year_pf) & (df_PF['месяц_digit'] <= month_pf)][['месяц','план','факт']].fillna(0)
result_ZP_AC = result_ZP_AC.rename(columns={'факт':f'факт_ЗП_АЦ_{year_pf}', 'план':f'план_ЗП_АЦ_{year_pf}'})
result_ZP_AS = df_PF[(df_PF['индекс'] == ind_pf_zp_AS )  & (df_PF['автоцентр'] == name_ac) &
                          (df_PF['год'] == year_pf) & (df_PF['месяц_digit'] <= month_pf)][['месяц','план','факт']].fillna(0)
result_ZP_AS = result_ZP_AS.rename(columns={'факт':f'факт_ЗП_АС_{year_pf}', 'план':f'план_ЗП_АС_{year_pf}'})
result_ZP_TC = df_PF[(df_PF['индекс'] == ind_pf_zp_TC )  & (df_PF['автоцентр'] == name_ac) &
                          (df_PF['год'] == year_pf) & (df_PF['месяц_digit'] <= month_pf)][['месяц','план','факт']].fillna(0)
result_ZP_TC = result_ZP_TC.rename(columns={'факт':f'факт_ЗП_ТЦ_{year_pf}', 'план':f'план_ЗП_ТЦ_{year_pf}'})

result_ZP_all = pd.merge(result_ZP_itog, result_ZP_AC, how='left')
result_ZP_all = pd.merge(result_ZP_all, result_ZP_AS, how='left')
result_ZP_all = pd.merge(result_ZP_all, result_ZP_TC, how='left')
df_DOX_ZP = pd.merge(result_doh_all, result_ZP_all, how='left')
df_DOX_ZP = convertor_zeros(df_DOX_ZP)


# блок с ИТОГОМ ЗП И ДОХОД
df_DOX_ZP[f'%_ФЗП_ИТОГО_ПЛАН_{year_pf}'] = round((df_DOX_ZP[f'план_ЗП_ИТОГО_{year_pf}'].astype(float) / df_DOX_ZP[f'план_дох_АЦ_{year_pf}'].astype(float) )*100, 2)
df_DOX_ZP[f'%_ФЗП_ИТОГО_ФАКТ_{year_pf}'] = round((df_DOX_ZP[f'факт_ЗП_ИТОГО_{year_pf}'].astype(float)  / df_DOX_ZP[f'факт_дох_АЦ_{year_pf}'].astype(float) )*100, 2)
df_DOX_ZP[f'ФЗП_ИТОГО_ВЫПОЛНЕНИЕ_{year_pf}'] = df_DOX_ZP[f'%_ФЗП_ИТОГО_ФАКТ_{year_pf}'].astype(float) - df_DOX_ZP[f'%_ФЗП_ИТОГО_ПЛАН_{year_pf}'].astype(float)

# блок с  ЗП И ДОХОД АС
df_DOX_ZP[f'%_ФЗП_АС_ПЛАН_{year_pf}'] = round((df_DOX_ZP[f'план_ЗП_АС_{year_pf}'].astype(float)  / df_DOX_ZP[f'план_дох_АС_{year_pf}'].astype(float) )*100, 2)
df_DOX_ZP[f'%_ФЗП_АС_ФАКТ_{year_pf}'] = round((df_DOX_ZP[f'факт_ЗП_АС_{year_pf}'].astype(float)  / df_DOX_ZP[f'факт_дох_АС_{year_pf}'].astype(float) )*100, 2)
df_DOX_ZP[f'ФЗП_АС_ВЫПОЛНЕНИЕ_{year_pf}'] = df_DOX_ZP[f'%_ФЗП_АС_ФАКТ_{year_pf}'].astype(float)  - df_DOX_ZP[f'%_ФЗП_АС_ПЛАН_{year_pf}'].astype(float)

# блок с  ЗП И ДОХОД ТЦ
df_DOX_ZP[f'%_ФЗП_ТЦ_ПЛАН_{year_pf}'] = round((df_DOX_ZP[f'план_ЗП_ТЦ_{year_pf}'].astype(float)  / df_DOX_ZP[f'план_дох_ТЦ_{year_pf}'].astype(float) )*100, 2)
df_DOX_ZP[f'%_ФЗП_ТЦ_ФАКТ_{year_pf}'] = round((df_DOX_ZP[f'факт_ЗП_ТЦ_{year_pf}'].astype(float)  / df_DOX_ZP[f'факт_дох_ТЦ_{year_pf}'].astype(float) )*100, 2)
df_DOX_ZP[f'ФЗП_ТЦ_ВЫПОЛНЕНИЕ_{year_pf}'] = df_DOX_ZP[f'%_ФЗП_ТЦ_ФАКТ_{year_pf}'].astype(float)  - df_DOX_ZP[f'%_ФЗП_ТЦ_ПЛАН_{year_pf}'].astype(float)

# блок с  ЗП подраздаелений от общего дохода
df_DOX_ZP[f'%_ФЗП_АЦ_ОТ_ОБЩ_ДОХ_{year_pf}'] = round((df_DOX_ZP[f'факт_ЗП_АЦ_{year_pf}'].astype(float)  / df_DOX_ZP[f'факт_дох_АЦ_{year_pf}'].astype(float) )*100, 2)
df_DOX_ZP[f'%_ФЗП_АС_ОТ_ОБЩ_ДОХ_{year_pf}'] = round((df_DOX_ZP[f'факт_ЗП_АС_{year_pf}'].astype(float)  / df_DOX_ZP[f'факт_дох_АЦ_{year_pf}'].astype(float) )*100, 2)
df_DOX_ZP[f'%_ФЗП_ТЦ_ОТ_ОБЩ_ДОХ_{year_pf}'] = round((df_DOX_ZP[f'факт_ЗП_ТЦ_{year_pf}'].astype(float)  / df_DOX_ZP[f'факт_дох_АЦ_{year_pf}'].astype(float) )*100, 2)

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(df_DOX_ZP['месяц'], df_DOX_ZP[f'%_ФЗП_ИТОГО_ПЛАН_{year_pf}'], label=f'%_ФЗП_ИТОГО_ПЛАН_{year_pf}',
             color='grey')
    plt.plot(df_DOX_ZP['месяц'], df_DOX_ZP[f'%_ФЗП_ИТОГО_ФАКТ_{year_pf}'], label=f'%_ФЗП_ИТОГО_ФАКТ_{year_pf}',
             color='cornflowerblue')
    # градус подписей по оси Х
    plt.xticks(rotation=0)
    plt.legend()
    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")
    plt.title(f"Доля ЗП АЦ\n к ДОХОДУ АЦ (П/Ф) \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig('Доля_ЗП_АЦ_1.png')

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(df_DOX_ZP['месяц'], df_DOX_ZP[f'%_ФЗП_АС_ПЛАН_{year_pf}'], label=f'%_ФЗП_АС_ПЛАН_{year_pf}', color='grey')
    plt.plot(df_DOX_ZP['месяц'], df_DOX_ZP[f'%_ФЗП_АС_ФАКТ_{year_pf}'], label=f'%_ФЗП_АС_ФАКТ_{year_pf}',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()

    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")
    plt.title(f"Динамика \n ФЗП АС (П/Ф) \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")

    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig('Динамика_ФЗП_АС.png')

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(df_DOX_ZP['месяц'], df_DOX_ZP[f'%_ФЗП_ТЦ_ПЛАН_{year_pf}'], label=f'%_ФЗП_ТЦ_ПЛАН_{year_pf}', color='grey')
    plt.plot(df_DOX_ZP['месяц'], df_DOX_ZP[f'%_ФЗП_ТЦ_ФАКТ_{year_pf}'], label=f'%_ФЗП_ТЦ_ФАКТ_{year_pf}',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()

    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"Динамика \n ФЗП ТЦ (П/Ф) \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig('Динамика_ФЗП_ТЦ.png')

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(df_DOX_ZP['месяц'], df_DOX_ZP[f'%_ФЗП_АЦ_ОТ_ОБЩ_ДОХ_{year_pf}'], label=f'АЦ', color='goldenrod')
    plt.plot(df_DOX_ZP['месяц'], df_DOX_ZP[f'%_ФЗП_АС_ОТ_ОБЩ_ДОХ_{year_pf}'], label=f'АС', color='cornflowerblue')
    plt.plot(df_DOX_ZP['месяц'], df_DOX_ZP[f'%_ФЗП_ТЦ_ОТ_ОБЩ_ДОХ_{year_pf}'], label=f'ТЦ', color='navy')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()

    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")
    plt.title(f"Доля ЗП подразделений \n от дохода \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")

    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig('Доля_зп_подразделенй_от_дохода.png')

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(df_DOX_ZP['месяц'], df_DOX_ZP[f'факт_дох_АЦ_{year_pf}'], label=f'Доход_ИТОГО', color='b')
    plt.plot(df_DOX_ZP['месяц'], df_DOX_ZP[f'факт_ЗП_ИТОГО_{year_pf}'], label=f'ЗП_ИТОГО', color='c')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()

    plt.ylabel('МЛН.РУБ', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"Доход и ЗП \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в млн.руб", fontsize=9,
              fontweight="bold")

    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig('Доход_к_зп_итого_.png')

# округлим значения в ЗП
for i in df_ZP.columns:
    try:
        df_ZP[i] = df_ZP[i].apply(lambda x: round(Decimal(x)))
    except:
        print(i)



doh_k_zp_itogo = pd.DataFrame({'Наименование': ['ФЗП_всего_к_дох_всего',
                                                'ФЗП_РУК_к_дох_всего',
                                                'ФЗП_АЦ_(без_РУК)_к_дох_всего',
                                                'ФЗП_ТЦ_к_дох_ТЦ',
                                                'ФЗП_АС_к_дох_АС'],
                    'Доход': [func_IND_podr(df_PF, ind_pf_doh),
                              func_IND_podr(df_PF, ind_pf_doh),
                              func_IND_podr(df_PF, ind_pf_doh),
                              func_IND_podr(df_PF, ind_pf_doh_tc),
                              func_IND_podr(df_PF, ind_pf_doh_as)
                              ],
                    'ФЗП': [func_ZP_podr_month(df_ZP),
                            func_ZP_podr_month(df_ZP, [], ['АЦ.Рук','АС.Рук','ТЦ.Рук']),
                            func_ZP_podr_month(df_ZP, ['АЦ'],['ФО','АЦ.общ']),
                            func_ZP_podr_month(df_ZP, ['ТЦ'], ['МЦ','ИТР.МЦ','ДО','ОЗЧ','КЦ','ИТР.КЦ']),
                            func_ZP_podr_month(df_ZP, ['АС'],['ОПА','АС.общ','ОВП']) ]
                    })

doh_k_zp_itogo = func_float_df_digit(doh_k_zp_itogo, float)
doh_k_zp_itogo['ФЗП_от_ДОХ_%'] = round(doh_k_zp_itogo['ФЗП']/doh_k_zp_itogo['Доход']*100, 2)
doh_k_zp_itogo = convertor_zeros(doh_k_zp_itogo, column=['Доход', 'ФЗП'])

doh_k_zp_itogo = func_float_df_digit(doh_k_zp_itogo, float)
doh_k_zp_itogo['ФЗП_от_ДОХ_%'] = round(doh_k_zp_itogo['ФЗП']/doh_k_zp_itogo['Доход']*100, 2)
doh_k_zp_itogo = convertor_zeros(doh_k_zp_itogo, column=['Доход', 'ФЗП'])

ttime_per = func_IND_podr(df_PF, ind_pf_doh_as) \
    + func_IND_podr(df_PF, om_ovp)\
    - func_IND_podr(df_PF, ind_pf_doh_ovp)\
    - func_IND_podr(df_PF, ind_pf_doh_prch_usl)\
    - func_IND_podr(df_PF, ind_pf_doh_otch_oup)


doh_k_zp_AS = pd.DataFrame({'Наименование': ['ФЗП_АСнов(отдел_ОПА)_к_дох_АСнов',
                                             'ФЗП_ОВП(отдел_ОВП)_к_дох_ОВП'],
                            'Доход':[ttime_per,
                                    func_IND_podr(df_PF, ind_pf_doh_as) - ttime_per],
                            'ФЗП': [func_ZP_podr_month(df_ZP, ['АС'], ['ОПА']) ,
                                    func_ZP_podr_month(df_ZP, ['АС'], ['ОВП'])]})

doh_k_zp_AS = func_float_df_digit(doh_k_zp_AS)

doh_k_zp_AS['ФЗП_от_ДОХ_%'] = round(doh_k_zp_AS['ФЗП']/doh_k_zp_AS['Доход']*100, 2)
del ttime_per
doh_k_zp_AS.fillna(0)
doh_k_zp_AS = convertor_zeros(doh_k_zp_AS, column=['Доход', 'ФЗП'])

doh_k_zp_TC = pd.DataFrame({'Наименование': ['ФЗП_МЦ_к_дох_МЦ',
                                             'ФЗП_КЦ_к_дох_КЦ',
                                             'ФЗП_ДО_к_дох_ДО'],
                            'Доход':[func_IND_podr(df_PF, ind_pf_doh_meh),
                                    func_IND_podr(df_PF, ind_pf_doh_kuz),
                                    func_IND_podr(df_PF, ind_pf_doh_do)],
                            'ФЗП': [func_ZP_podr_month(df_ZP, ['ТЦ'],['МЦ', 'ИТР.МЦ']),
                                    func_ZP_podr_month(df_ZP, ['ТЦ'],['КЦ', 'ИТР.КЦ']),
                                    func_ZP_podr_month(df_ZP, ['ТЦ'],['ДО'])
                                    ]})


doh_k_zp_TC = func_float_df_digit(doh_k_zp_TC)

doh_k_zp_TC['ФЗП_от_ДОХ_%'] = round(doh_k_zp_TC['ФЗП']/doh_k_zp_TC['Доход']*100, 2)
doh_k_zp_TC = doh_k_zp_TC.fillna(0)
doh_k_zp_TC = convertor_zeros(doh_k_zp_TC, column=['Доход', 'ФЗП'])

doh_k_zp_MC = pd.DataFrame({'Наименование': ['ФЗП_МЦ_к_дох_МЦ', 'ФЗП_механиков_к_дох_МЦ', 'ФЗП_ИТР_МЦ_к_дох_МЦ'],
                            'Доход':[func_IND_podr(df_PF, ind_pf_doh_meh),
                                     func_IND_podr(df_PF, ind_pf_doh_meh),
                                     func_IND_podr(df_PF, ind_pf_doh_meh)],
                            'ФЗП': [func_ZP_podr_month(df_ZP, ['ТЦ'], ['МЦ','ИТР.МЦ']),
                                    func_ZP_podr_month(df_ZP, ['ТЦ'], ['МЦ']),
                                    func_ZP_podr_month(df_ZP, ['ТЦ'], ['ИТР.МЦ'])
                                    ]})

doh_k_zp_MC = func_float_df_digit(doh_k_zp_MC)
doh_k_zp_MC['ФЗП_от_ДОХ_%'] = round(doh_k_zp_MC['ФЗП']/doh_k_zp_MC['Доход']*100, 2)
doh_k_zp_MC = convertor_zeros(doh_k_zp_MC, column=['Доход', 'ФЗП'])

doh_k_zp_KC = pd.DataFrame({'Наименование': ['ФЗП_КЦ_к_дох_КЦ',
                                             'ФЗП_механиков_к_дох_КЦ',
                                             'ФЗП_ИТР_МЦ_к_дох_КЦ'],
                            'Доход':[func_IND_podr(df_PF, ind_pf_doh_kuz),
                                     func_IND_podr(df_PF, ind_pf_doh_kuz),
                                     func_IND_podr(df_PF, ind_pf_doh_kuz)],
                            'ФЗП': [func_ZP_podr_month(df_ZP, ['ТЦ'], ['КЦ','ИТР.КЦ']),
                                    func_ZP_podr_month(df_ZP, ['ТЦ'], ['КЦ']),
                                    func_ZP_podr_month(df_ZP, ['ТЦ'], ['ИТР.КЦ'])
                                    ]})

doh_k_zp_KC = func_float_df_digit(doh_k_zp_KC)
doh_k_zp_KC['ФЗП_от_ДОХ_%'] = round(doh_k_zp_KC['ФЗП']/doh_k_zp_KC['Доход']*100, 2)
doh_k_zp_KC = doh_k_zp_KC.fillna(0)
doh_k_zp_KC = convertor_zeros(doh_k_zp_KC, column=['Доход', 'ФЗП'])


doh_itog_k_zp_itog = Indicators('ЗП_итого_к_Доходу_итого', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh, 'факт'),
                        func_ZP_podr_all_year(df_ZP, [], []), how='left'))
doh_itog_k_zp_itog.rename()
convertor_zeros(doh_itog_k_zp_itog.df) # в самом классе df не конверитирован в 0.000 три знака после запятой


doh_itog_k_zp_ruk = Indicators('Доход_итого_к_ЗП_РУК', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh, 'факт'),
                        func_ZP_podr_all_year(df_ZP, [], ['АЦ.Рук','АС.Рук','ТЦ.Рук']), how='left'))
doh_itog_k_zp_ruk.rename()
convertor_zeros(doh_itog_k_zp_ruk.df)


doh_itog_k_zp_AC = Indicators('Доход_итого_к_ЗП_АЦ', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh, 'факт'),
                        func_ZP_podr_all_year(df_ZP, ['АЦ'], ['ФО','АЦ.общ']), how='left'))
doh_itog_k_zp_AC.rename()
convertor_zeros(doh_itog_k_zp_AC.df)


doh_TC_k_zp_TC = Indicators('ЗП_ТЦ_к_Доходу_ТЦ', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh_tc, 'факт'),
                        func_ZP_podr_all_year(df_ZP, ['ТЦ'], ['МЦ','ИТР.МЦ','ДО','ОЗЧ','КЦ','ИТР.КЦ']), how='left'))
doh_TC_k_zp_TC.rename()
convertor_zeros(doh_TC_k_zp_TC.df)


doh_AS_k_zp_AS = Indicators('ЗП_АС_общая_к_Доходу_АС_общему', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh_as, 'факт'),
                        func_ZP_podr_all_year(df_ZP, ['АС'], ['ОПА','АС.общ','ОВП']), how='left'))
doh_AS_k_zp_AS.rename()
convertor_zeros(doh_AS_k_zp_AS.df)

# доход АС новых авто и ОВП нужно разделить собираем по новым авто
res_dhs = func_IND_podr_year(df_PF, ind_pf_doh_as, 'факт')
res_dhs = res_dhs.rename(columns={res_dhs.columns[-1]: f"{ind_pf_doh_as}"})

res_om_ovp = func_IND_podr_year(df_PF, om_ovp, 'факт')
res_om_ovp = res_om_ovp.rename(columns={res_om_ovp.columns[-1]: f"{om_ovp}"})

res_dh_ovp = func_IND_podr_year(df_PF, ind_pf_doh_ovp, 'факт')
res_dh_ovp = res_dh_ovp.rename(columns={res_dh_ovp.columns[-1]: f"{ind_pf_doh_ovp}"})

res_dh_prch_ul = func_IND_podr_year(df_PF, ind_pf_doh_prch_usl, 'факт')
res_dh_prch_ul = res_dh_prch_ul.rename(columns={res_dh_prch_ul.columns[-1]: f"{ind_pf_doh_prch_usl}"})

res_dh_otch_ovp = func_IND_podr_year(df_PF, ind_pf_doh_otch_oup, 'факт')
res_dh_otch_ovp = res_dh_otch_ovp.rename(columns={res_dh_otch_ovp.columns[-1]: f"{ind_pf_doh_otch_oup}"})

res_time = pd.merge(res_dhs, res_om_ovp, how='left')
for i in [res_dh_ovp, res_dh_prch_ul, res_dh_otch_ovp]:
    res_time = pd.merge(res_time, i, how='left')

res_time['факт_дох_АСнов'] = res_time['220'] + res_time['1513'] - res_time['БуДох'] - res_time['1420'] - res_time[
    '1514']
res_time = res_time[['месяц', 'факт_дох_АСнов']]
# res_time

# отдельно собираем ЗП АС новых ам
res_zp_as_n = func_ZP_podr_all_year(df_ZP, ['АС'], ['ОПА'])

doh_ASnew_k_zp_ASnew = Indicators('ЗП_АС_нов_к_Доходу_АС_нов', 'итого',
               pd.merge(res_time,
                        res_zp_as_n, how='left'))
doh_ASnew_k_zp_ASnew.rename()
convertor_zeros(doh_ASnew_k_zp_ASnew.df)

# собираем доход овп
res_doh_as_ovp =  pd.merge(func_IND_podr_year(df_PF, ind_pf_doh_as, 'факт'), res_time, how='left')
res_doh_as_ovp['доход_ОВП'] = res_doh_as_ovp[f'факт_дох_{year_pf}'] - res_doh_as_ovp['факт_дох_АСнов']
res_doh_as_ovp = res_doh_as_ovp[['месяц','доход_ОВП']]
#res_doh_as_ovp
doh_AS_k_zp_AS_ovp = Indicators('ЗП_АС_овп_к_Доходу_АС_овп', 'итого',
               pd.merge(res_doh_as_ovp,
                        func_ZP_podr_all_year(df_ZP, ['АС'], ['ОВП']), how='left').fillna(0))
doh_AS_k_zp_AS_ovp.rename()
convertor_zeros(doh_AS_k_zp_AS_ovp.df)


doh_MC_k_zp_MC = Indicators('ЗП_МЦ_к_Доходу_МЦ', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh_meh, 'факт'),
                        func_ZP_podr_all_year(df_ZP, ['ТЦ'], ['МЦ', 'ИТР.МЦ']), how='left'))
doh_MC_k_zp_MC.rename()
convertor_zeros(doh_MC_k_zp_MC.df)


doh_MC_k_zp_MC_meh = Indicators('ЗП_мех_МЦ_к_Доходу_МЦ', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh_meh, 'факт'),
                        func_ZP_podr_all_year(df_ZP, ['ТЦ'], ['МЦ']), how='left'))
doh_MC_k_zp_MC_meh.rename()
convertor_zeros(doh_MC_k_zp_MC_meh.df)


doh_MC_k_zp_MC_itr = Indicators('ЗП_итр_МЦ_к_Доходу_МЦ', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh_meh, 'факт'),
                        func_ZP_podr_all_year(df_ZP, ['ТЦ'], ['ИТР.МЦ']), how='left'))
doh_MC_k_zp_MC_itr.rename()
convertor_zeros(doh_MC_k_zp_MC_itr.df)


doh_KC_k_zp_KC = Indicators('ЗП_КЦ_к_Доходу_КЦ', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh_kuz, 'факт'),
                        func_ZP_podr_all_year(df_ZP, ['ТЦ'], ['КЦ', 'ИТР.КЦ']), how='left').fillna(0))
doh_KC_k_zp_KC.rename()
convertor_zeros(doh_KC_k_zp_KC.df)


doh_KC_k_zp_KC_meh = Indicators('ЗП_мех_КЦ_к_Доходу_КЦ', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh_kuz, 'факт'),
                        func_ZP_podr_all_year(df_ZP, ['ТЦ'], ['КЦ']), how='left').fillna(0))
doh_KC_k_zp_KC_meh.rename()
convertor_zeros(doh_KC_k_zp_KC_meh.df)


doh_KC_k_zp_KC_itr = Indicators('ЗП_итр_КЦ_к_Доходу_КЦ', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh_kuz, 'факт'),
                        func_ZP_podr_all_year(df_ZP, ['ТЦ'], ['ИТР.КЦ']), how='left').fillna(0))
doh_KC_k_zp_KC_itr.rename()
convertor_zeros(doh_KC_k_zp_KC_itr.df)


doh_DO_k_zp_DO = Indicators('ЗП_ДО_к_Доходу_ДО', 'итого',
               pd.merge(func_IND_podr_year(df_PF, ind_pf_doh_do, 'факт'),
                        func_ZP_podr_all_year(df_ZP, ['ТЦ'], ['ДО']), how='left'))
doh_DO_k_zp_DO.rename()
convertor_zeros(doh_DO_k_zp_DO.df)

# Численность ТЦ
pers_count_TC = func_ZP_count_pers(df_ZP, month_pf, ['ТЦ'], [])
# Численность МЦ
pers_count_MC = func_ZP_count_pers(df_ZP, month_pf, [], ['МЦ', 'ИТР.МЦ'])
# Численность КЦ
pers_count_KC = func_ZP_count_pers(df_ZP, month_pf, [], ['КЦ', 'ИТР.КЦ'])
# Численность ДО
pers_count_DO = func_ZP_count_pers(df_ZP, month_pf, [], ['ДО', 'ИТР.ДО'])
# Численность прочие
pers_count_proch = np.array(func_ZP_count_pers(df_ZP, month_pf, ['ТЦ'], [])) - \
                   np.array(func_ZP_count_pers(df_ZP, month_pf, [], ['МЦ', 'ИТР.МЦ'])) - \
                   np.array(func_ZP_count_pers(df_ZP, month_pf, [], ['КЦ', 'ИТР.КЦ'])) - \
                   np.array(func_ZP_count_pers(df_ZP, month_pf, [], ['ДО', 'ИТР.ДО']))

# Численность МЦ производственный персонал
pers_count_MC_proizv = func_ZP_count_pers(df_ZP, month_pf, [], ['МЦ'])
# Численность КЦ производственный персонал
pers_count_KC_proizv = func_ZP_count_pers(df_ZP, month_pf, [], ['КЦ'])

# ОТКУДА БРАТЬ СРЕДНИЙ ЧЕК ???

month_ = [spravka_month.get(i) for i in
          df_ZP[(df_ZP['Год'] == year_pf) & (df_ZP['Месяц'] <= month_pf)]['Месяц'].unique()]
df_test = pd.DataFrame({'месяц': month_,
                        'Численность_ТЦ': pers_count_TC,
                        'Численность_МЦ': pers_count_MC,
                        'Численность_КЦ': pers_count_KC,
                        'Численность_ДО': pers_count_DO,
                        'Численность_проч': pers_count_proch,
                        'в_тч_Численность_МЦ_произв_перс': pers_count_MC_proizv,
                        'в_тч_Численность_КЦ_произв_перс': pers_count_KC_proizv,
                        'нч_механический': list(func_IND_podr_year(df_PF, ind_pf_nch_meh).iloc[:, 1]),
                        'в_тч_ТОиР': list(func_IND_podr_year(df_PF, ind_pf_nch_TO_R).iloc[:, 1]),
                        'нч_кузовной': list(func_IND_podr_year(df_PF, ind_pf_nch_kuz).iloc[:, 1]),
                        'нч_до': list(func_IND_podr_year(df_PF, ind_pf_nch_do).iloc[:, 1]),

                        'ЭТОТ БЛОК ВЫРУЧКИ СКРЫТЬ:': [' ----- ' for i in range(month_pf)],

                        'Выручка_ТЦ': list(func_IND_podr_year(df_PF, ind_pf_revenue_mex_itg).iloc[:, 1]),
                        'Выручка_ТЦ_МЕХ_работы': list(func_IND_podr_year(df_PF, ind_pf_revenue_mex_rab).iloc[:, 1]),
                        'Выручка_ТЦ_МЕХ_запчасти': list(func_IND_podr_year(df_PF, ind_pf_revenue_mex_zch).iloc[:, 1]),
                        'Выручка_ТЦ_КУЗ_работы': list(func_IND_podr_year(df_PF, ind_pf_revenue_kuz_rab).iloc[:, 1]),
                        'Выручка_ТЦ_КУЗ_запчасти': list(func_IND_podr_year(df_PF, ind_pf_revenue_kuz_zch).iloc[:, 1]),
                        'Выручка_ТЦ_ДО_работы': list(func_IND_podr_year(df_PF, ind_pf_revenue_do_rab).iloc[:, 1]),
                        'Выручка_ТЦ_ДО_запчасти': list(func_IND_podr_year(df_PF, ind_pf_revenue_do_zch).iloc[:, 1]),
                        'Выручка_ТЦ_продажа_запчасти': list(
                            func_IND_podr_year(df_PF, ind_pf_revenue_real_zch).iloc[:, 1]),
                        'Выручка_ТЦ_прочее': list(func_IND_podr_year(df_PF, ind_pf_revenue_proch).iloc[:, 1]),
                        'Выручка_Бонус_ТЦ': list(func_IND_podr_year(df_PF, ind_pf_revenue_bonus_TC).iloc[:, 1]),
                        'Выручка_ТЦ_прочие_услуги': list(
                            func_IND_podr_year(df_PF, ind_pf_revenue_proch_usl).iloc[:, 1]),

                        'ЭТОТ БЛОК ДОХОДА СКРЫТЬ:': [' ----- ' for i in range(month_pf)],

                        'Доход_ТЦ': np.array(list(func_IND_podr_year(df_PF, ind_pf_doh_tc).iloc[:, 1])) + np.array(
                            list(func_IND_podr_year(df_PF, ind_pf_doh_bonus_program).iloc[:, 1])),
                        'Доход_ТЦ_МЕХ_работы': list(func_IND_podr_year(df_PF, ind_pf_doh_meh_rab).iloc[:, 1]),
                        'Доход_ТЦ_МЕХ_запчасти': list(func_IND_podr_year(df_PF, ind_pf_doh_meh_zch).iloc[:, 1]),
                        'Доход_ТЦ_КУЗ_работы': list(func_IND_podr_year(df_PF, ind_pf_doh_kuz_rab).iloc[:, 1]),
                        'Доход_ТЦ_КУЗ_запчасти': list(func_IND_podr_year(df_PF, ind_pf_doh_kuz_zch).iloc[:, 1]),
                        'Доход_ТЦ_ДО_работы': list(func_IND_podr_year(df_PF, ind_pf_doh_do_rab).iloc[:, 1]),
                        'Доход_ТЦ_ДО_запчасти': list(func_IND_podr_year(df_PF, ind_pf_doh_do_zch).iloc[:, 1]),
                        'Доход_ТЦ_продажа_запчасти': list(func_IND_podr_year(df_PF, ind_pf_doh_real_zch).iloc[:, 1]),
                        'Доход_ТЦ_прочее': list(func_IND_podr_year(df_PF, ind_pf_doh_proch_).iloc[:, 1]),
                        'Доход_Бонус_ТЦ': list(func_IND_podr_year(df_PF, ind_pf_doh_bonus).iloc[:, 1]),
                        'Доход_ТЦ_прочие_услуги': list(func_IND_podr_year(df_PF, ind_pf_doh_prch_usl_TC).iloc[:, 1]),
                        'Доход_ТЦ_бонусная_программа': list(
                            func_IND_podr_year(df_PF, ind_pf_doh_bonus_program).iloc[:, 1]),
                        'БП_ТЦ': list(func_IND_podr_year(df_PF, ind_pf_balance_profit).iloc[:, 1])

                        })
#df_test
# после транспонируем
df_test.T



df_test['БЛОК_ВЫРУЧКИ :'] = [' ----- ' for i in range(month_pf)]

df_test['Выручка_на_1_сотр'] = round(df_test['Выручка_ТЦ']/df_test['Численность_ТЦ'],1)

# блок выручки МЦ
df_test['БЛОК_ВЫРУЧКИ МЦ :'] = [' ----- ' for i in range(month_pf)]
df_test['Выручка_МЦ_Раб+ЗЧ_на_1_сотр'] = round((df_test['Выручка_ТЦ_МЕХ_работы']+
                                          df_test['Выручка_ТЦ_МЕХ_запчасти']+
                                          df_test['Выручка_ТЦ_продажа_запчасти']+
                                          df_test['Выручка_ТЦ_прочее'])/df_test['Численность_МЦ'],1)
df_test['Выручка_МЦ_Раб+ЗЧ'] = round((df_test['Выручка_ТЦ_МЕХ_работы']+
                                          df_test['Выручка_ТЦ_МЕХ_запчасти']+
                                          df_test['Выручка_ТЦ_продажа_запчасти']+
                                          df_test['Выручка_ТЦ_прочее']),1)
df_test['Выручка_работы_МЦ'] = round(df_test['Выручка_ТЦ_МЕХ_работы'],1)
df_test['Выручка_запчасти_МЦ'] = round((df_test['Выручка_ТЦ_МЕХ_запчасти']+
                                          df_test['Выручка_ТЦ_продажа_запчасти']+
                                          df_test['Выручка_ТЦ_прочее']),1)
df_test['Выручка_прочее_МЦ'] = round(df_test['Выручка_Бонус_ТЦ'] + df_test['Выручка_ТЦ_прочие_услуги'],1)

# блок выручки КЦ
df_test['БЛОК_ВЫРУЧКИ КЦ :'] = [' ----- ' for i in range(month_pf)]
df_test['Выручка_КЦ_Раб+ЗЧ_на_1_сотр'] = round((df_test['Выручка_ТЦ_КУЗ_работы']+
                                          df_test['Выручка_ТЦ_КУЗ_запчасти'])/df_test['Численность_КЦ'],1)
df_test['Выручка_КЦ_Раб+ЗЧ'] = round((df_test['Выручка_ТЦ_КУЗ_работы']+
                                          df_test['Выручка_ТЦ_КУЗ_запчасти']),1)
df_test['Выручка_работы_КЦ'] = round(df_test['Выручка_ТЦ_КУЗ_работы'],1)
df_test['Выручка_запчасти_КЦ'] = round(df_test['Выручка_ТЦ_КУЗ_запчасти'],1)
df_test['Выручка_прочее_КЦ'] = round(0,1)

# блок выручки ДО
df_test['БЛОК_ВЫРУЧКИ ДО :'] = [' ----- ' for i in range(month_pf)]
df_test['Выручка_ДО_Раб+ЗЧ_на_1_сотр'] = round((df_test['Выручка_ТЦ_ДО_работы']+
                                          df_test['Выручка_ТЦ_ДО_запчасти'])/df_test['Численность_ДО'],1)
df_test['Выручка_ДО_Раб+ЗЧ'] = round((df_test['Выручка_ТЦ_ДО_работы']+
                                          df_test['Выручка_ТЦ_ДО_запчасти']),1)
df_test['Выручка_работы_ДО'] = round(df_test['Выручка_ТЦ_ДО_работы'],1)
df_test['Выручка_запчасти_ДО'] = round(df_test['Выручка_ТЦ_ДО_запчасти'],1)
df_test['Выручка_прочее_ДО'] = round(0,1)

# блок дохода ТЦ
df_test['БЛОК_ДОХОДА_ТЦ_МЕХ:'] = [' ----- ' for i in range(month_pf)]
df_test['Доход_МЦ_раб_и_зч_на_1_сотр_пр_перс'] = round((df_test['Доход_ТЦ_МЕХ_работы'] +df_test['Доход_ТЦ_МЕХ_запчасти']) / df_test['в_тч_Численность_МЦ_произв_перс'],1)
df_test['Доход_МЦ_работы_на_1_сотр_пр_перс'] = round(df_test['Доход_ТЦ_МЕХ_работы'] / df_test['в_тч_Численность_МЦ_произв_перс'],1)
df_test['Доход_МЦ_запчасти_на_1_сотр_пр_перс'] = round((df_test['Доход_ТЦ_МЕХ_запчасти'] + df_test['Доход_ТЦ_продажа_запчасти'] + df_test['Доход_ТЦ_прочее']) / df_test['в_тч_Численность_МЦ_произв_перс'],1)
df_test['Доход_ТЦ_на_1_сотр'] = round(df_test['Доход_ТЦ'] / df_test['Численность_ТЦ'],1)

df_test['Доход_МЦ_Раб+ЗЧ'] = round((df_test['Доход_ТЦ_МЕХ_работы']+
                                          df_test['Доход_ТЦ_МЕХ_запчасти']+
                                          df_test['Доход_ТЦ_продажа_запчасти']+
                                          df_test['Доход_ТЦ_прочее']),1)

df_test['Доход_работы_МЦ'] = round(df_test['Доход_ТЦ_МЕХ_работы'],1)
df_test['Доход_запчасти_МЦ'] = round((df_test['Доход_ТЦ_МЕХ_запчасти']+
                                          df_test['Доход_ТЦ_продажа_запчасти']+
                                          df_test['Доход_ТЦ_прочее']),1)
df_test['Доход_прочее_МЦ'] = round(df_test['Доход_Бонус_ТЦ'] + df_test['Доход_ТЦ_прочие_услуги'],1)

# блок дохода КЦ
df_test['БЛОК_ДОХОДА_КЦ:'] = [' ----- ' for i in range(month_pf)]

df_test['Доход_КЦ_Раб+ЗЧ_на_1_сотр'] = round((df_test['Доход_ТЦ_КУЗ_работы']+
                                          df_test['Доход_ТЦ_КУЗ_запчасти'])/df_test['Численность_КЦ'],1)
df_test['Доход_КЦ_Раб+ЗЧ'] = round((df_test['Доход_ТЦ_КУЗ_работы']+
                                          df_test['Доход_ТЦ_КУЗ_запчасти']),1)
df_test['Доход_работы_КЦ'] = round(df_test['Доход_ТЦ_КУЗ_работы'],1)
df_test['Доход_запчасти_КЦ'] = round(df_test['Доход_ТЦ_КУЗ_запчасти'],1)
df_test['Доход_прочее_КЦ'] = round(0,1)

# блок дохода ДО
df_test['БЛОК_ДОХОДА_ДО :'] = [' ----- ' for i in range(month_pf)]
df_test['Доход_ДО_Раб+ЗЧ_на_1_сотр'] = round((df_test['Доход_ТЦ_ДО_работы']+
                                          df_test['Доход_ТЦ_ДО_запчасти'])/df_test['Численность_ДО'],1)
df_test['Доход_ДО_Раб+ЗЧ'] = round((df_test['Доход_ТЦ_ДО_работы']+
                                          df_test['Доход_ТЦ_ДО_запчасти']),1)
df_test['Доход_работы_ДО'] = round(df_test['Доход_ТЦ_ДО_работы'],1)
df_test['Доход_запчасти_ДО'] = round(df_test['Доход_ТЦ_ДО_запчасти'],1)
df_test['Доход_прочее_ДО'] = round(0,1)

df_test['БП_ТЦ_на_1_сотр'] = round(df_test['БП_ТЦ'] / df_test['Численность_ТЦ'],1)
df_test = df_test.apply(infinity_).fillna(0)
#df_test
#df_test.T
convertor_zeros(df_test, column=['БЛОК_ВЫРУЧКИ :', 'Выручка_на_1_сотр', 'Выручка_МЦ_Раб+ЗЧ_на_1_сотр',
                                 'Выручка_МЦ_Раб+ЗЧ', 'Выручка_работы_МЦ', 'Выручка_запчасти_МЦ', 'Выручка_прочее_МЦ',
                                 'БЛОК_ВЫРУЧКИ КЦ :', 'Выручка_КЦ_Раб+ЗЧ_на_1_сотр', 'Выручка_КЦ_Раб+ЗЧ',
                                 'Выручка_работы_КЦ', 'Выручка_запчасти_КЦ', 'Выручка_прочее_КЦ', 'БЛОК_ВЫРУЧКИ ДО :',
                                 'Выручка_ДО_Раб+ЗЧ_на_1_сотр', 'Выручка_ДО_Раб+ЗЧ', 'Выручка_работы_ДО',
                                 'Выручка_запчасти_ДО', 'Выручка_прочее_ДО',

                                 'БЛОК_ДОХОДА_ТЦ_МЕХ:', 'Доход_МЦ_раб_и_зч_на_1_сотр_пр_перс',
                                 'Доход_МЦ_работы_на_1_сотр_пр_перс', 'Доход_МЦ_запчасти_на_1_сотр_пр_перс',
                                 'Доход_ТЦ_на_1_сотр_пр_перс', 'Доход_ТЦ_на_1_сотр', 'Доход_МЦ_Раб+ЗЧ',
                                 'Доход_работы_МЦ', 'Доход_запчасти_МЦ', 'Доход_прочее_МЦ', 'БЛОК_ДОХОДА_КЦ:',
                                 'Доход_КЦ_Раб+ЗЧ_на_1_сотр', 'Доход_КЦ_Раб+ЗЧ', 'Доход_работы_КЦ',
                                 'Доход_запчасти_КЦ', 'Доход_прочее_КЦ', 'БЛОК_ДОХОДА_ДО :',
                                 'Доход_ДО_Раб+ЗЧ_на_1_сотр', 'Доход_ДО_Раб+ЗЧ', 'Доход_работы_ДО', 'Доход_запчасти_ДО',
                                 'Доход_прочее_ДО', 'БП_ТЦ', 'БП_ТЦ_на_1_сотр'])

# df_test.columns
lst_dopusk_rev = ('месяц', 'Численность_ТЦ', 'Численность_МЦ', 'Численность_КЦ', 'Численность_ДО', 'Численность_проч', 'в_тч_Численность_МЦ_произв_перс',
              'в_тч_Численность_КЦ_произв_перс', 'нч_механический', 'в_тч_ТОиР', 'нч_кузовной', 'нч_до', 'БЛОК_ВЫРУЧКИ :', 'Выручка_на_1_сотр', 'Выручка_МЦ_Раб+ЗЧ_на_1_сотр',
              'Выручка_МЦ_Раб+ЗЧ', 'Выручка_работы_МЦ', 'Выручка_запчасти_МЦ', 'Выручка_прочее_МЦ', 'БЛОК_ВЫРУЧКИ КЦ :', 'Выручка_КЦ_Раб+ЗЧ_на_1_сотр', 'Выручка_КЦ_Раб+ЗЧ',
              'Выручка_работы_КЦ', 'Выручка_запчасти_КЦ', 'Выручка_прочее_КЦ', 'БЛОК_ВЫРУЧКИ ДО :',	'Выручка_ДО_Раб+ЗЧ_на_1_сотр', 'Выручка_ДО_Раб+ЗЧ', 'Выручка_работы_ДО',
              'Выручка_запчасти_ДО', 'Выручка_прочее_ДО')
res_x_vr = df_test[[i for i in df_test.columns if i in lst_dopusk_rev]].T

res_x_vr.columns = res_x_vr.iloc[0]
res_x_vr = res_x_vr.iloc[1:]

lst_dopusk_doh = ('месяц', 'Численность_ТЦ', 'Численность_МЦ', 'Численность_КЦ', 'Численность_ДО', 'Численность_проч', 'в_тч_Численность_МЦ_произв_перс',
              'в_тч_Численность_КЦ_произв_перс', 'нч_механический', 'в_тч_ТОиР', 'нч_кузовной', 'нч_до', 'БЛОК_ДОХОДА_ТЦ_МЕХ:', 'Доход_МЦ_раб_и_зч_на_1_сотр_пр_перс',
              'Доход_МЦ_работы_на_1_сотр_пр_перс', 'Доход_МЦ_запчасти_на_1_сотр_пр_перс', 'Доход_ТЦ_на_1_сотр_пр_перс', 'Доход_ТЦ_на_1_сотр', 'Доход_МЦ_Раб+ЗЧ',
              'Доход_работы_МЦ', 'Доход_запчасти_МЦ', 'Доход_прочее_МЦ', 'БЛОК_ДОХОДА_КЦ:', 'Доход_КЦ_Раб+ЗЧ_на_1_сотр', 'Доход_КЦ_Раб+ЗЧ', 'Доход_работы_КЦ',
              'Доход_запчасти_КЦ', 'Доход_прочее_КЦ', 'БЛОК_ДОХОДА_ДО :', 'Доход_ДО_Раб+ЗЧ_на_1_сотр', 'Доход_ДО_Раб+ЗЧ', 'Доход_работы_ДО', 'Доход_запчасти_ДО',
              'Доход_прочее_ДО', 'БП_ТЦ', 'БП_ТЦ_на_1_сотр')


res_x_doh = df_test[[i for i in df_test.columns if i in lst_dopusk_doh]].T
res_x_doh.columns = res_x_doh.iloc[0]
res_x_doh = res_x_doh.iloc[1:]


with open("process_progress.txt", "w") as file:  # для заполнения прогрессбара
    file.write("50")


dfi.export(df_last_first_target_x, 'df_styled.png', dpi=150)

fig = df2img.plot_dataframe(
    df_table_1,
    print_index=False,  # отключает индексы
    col_width=[0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.5],  # ширина колонок
    title=dict(
        font_color="black",
        font_family="Times New Roman",
        font_size=20,
        text=f"Выполнение распоряжения оптимизации расходов за {spravka_month.get(month_pf)}",
        x=0,
        xanchor="left",
    ),
    tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=14,
        height=20,
        line_width=5,
    ),
    tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=5
    ),
    fig_size=(1100, 150),
)
df2img.save_dataframe(fig=fig, filename="df_распоряжение_оптимизации.png")


fig = df2img.plot_dataframe(
      OM_rspred_tabl.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.1, 0.12, 0.11, 0.13, 0.1, 0.1, 0.2, 0.13, 0.25, 0.15],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text="Распределение доходности ОМ согласно FR, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=14,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_распределение_дох_ом.png")

fig = df2img.plot_dataframe(OM_rspred_nacopl.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.1, 0.12, 0.11, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text="OМ и авто накопительно согласно FR, (млн.руб) / (ам шт.)",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=14,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_распределение_дох_ом_накопительно.png")

fig = df2img.plot_dataframe(df_DOX_ZP[['месяц',
                                     f'план_дох_АЦ_{year_pf}',f'факт_дох_АЦ_{year_pf}',
                                     f'план_ЗП_ИТОГО_{year_pf}', f'факт_ЗП_ИТОГО_{year_pf}',
                                    f'%_ФЗП_ИТОГО_ПЛАН_{year_pf}',	f'%_ФЗП_ИТОГО_ФАКТ_{year_pf}']].astype(str),
      print_index=False, # отключает индексы
      col_width=[0.1, 0.15, 0.15, 0.17, 0.17, 0.2, 0.2],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text="ЗП всего автоцентра (по FR) к ДОХОДУ всего автоцентра (по FR), млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=11,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доля_зп_к_доходу_итого.png")

fig = df2img.plot_dataframe(df_DOX_ZP[['месяц',
           f'план_дох_АС_{year_pf}', f'факт_дох_АС_{year_pf}',
           f'план_ЗП_АС_{year_pf}', f'факт_ЗП_АС_{year_pf}',
           f'%_ФЗП_АС_ПЛАН_{year_pf}',f'%_ФЗП_АС_ФАКТ_{year_pf}']].astype(str),
      print_index=False, # отключает индексы
      col_width=[0.1, 0.15, 0.15, 0.17, 0.17, 0.2, 0.2],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text="Динамика ФЗП АС (по FR) к доходу АС (по FR), млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=11,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_динамика_фзп_ас.png")

fig = df2img.plot_dataframe(df_DOX_ZP[['месяц',
           f'план_дох_ТЦ_{year_pf}', f'факт_дох_ТЦ_{year_pf}',
           f'план_ЗП_ТЦ_{year_pf}', f'факт_ЗП_ТЦ_{year_pf}',
           f'%_ФЗП_ТЦ_ПЛАН_{year_pf}',f'%_ФЗП_ТЦ_ФАКТ_{year_pf}']].astype(str),
      print_index=False, # отключает индексы
      col_width=[0.1, 0.15, 0.15, 0.17, 0.17, 0.2, 0.2],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text="Динамика ФЗП ТЦ (по FR) к доходу ТЦ (по FR), млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=11,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_динамика_фзп_тц.png")

fig = df2img.plot_dataframe(df_DOX_ZP[['месяц',
           f'%_ФЗП_АЦ_ОТ_ОБЩ_ДОХ_{year_pf}', f'%_ФЗП_АС_ОТ_ОБЩ_ДОХ_{year_pf}',
           f'%_ФЗП_ТЦ_ОТ_ОБЩ_ДОХ_{year_pf}']].astype(str),
      print_index=False, # отключает индексы
      col_width=[0.1, 0.15, 0.15, 0.17, 0.17, 0.2, 0.2],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text="Доля ЗП подразделений от дохода автоцентра (по FR), млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=11,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доля_зп_подр_от_дох.png")

fig = df2img.plot_dataframe(df_DOX_ZP[['месяц',
           f'факт_дох_АЦ_{year_pf}', f'факт_ЗП_ИТОГО_{year_pf}']].apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.1, 0.15, 0.15, 0.17, 0.17, 0.2, 0.2],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text="Доход и ЗП всего (по FR), млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=11,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_к_зп_итого.png")

dfi.export(func_limit_ZP(df_ZP, limit=lim_zp_max, podr=['АЦ']), 'df_лимит_зп_ац.png', dpi=150)
dfi.export(func_limit_ZP(df_ZP, limit=lim_zp_max, podr=['АС']), 'df_лимит_зп_ас.png', dpi=150)
dfi.export(func_limit_ZP(df_ZP, limit=lim_zp_max, podr=['ТЦ']), 'df_лимит_зп_тц.png', dpi=150)

fig = df2img.plot_dataframe(doh_k_zp_itogo.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП итого (по ведомости) к Доходу итого (по FR) в разбивке за {spravka_month.get(month_pf)}, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 235),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_итого_к_зп_по_ведомости_итого.png")

fig = df2img.plot_dataframe(
      func_float_df_digit(doh_k_zp_AS.fillna(0)),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП АС (по ведомости) (АСновые/ОВП) без РУК АС к Доходу АС (по FR) в разбивке за {spravka_month.get(month_pf)}, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 140),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_ас_к_зп_ас_ведомость.png")

fig = df2img.plot_dataframe(doh_k_zp_TC.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП ТЦ (по ведомости) к Доходу ТЦ (по FR) за {spravka_month.get(month_pf)} в разбивке, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 170),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_тц_к_зп_тц_ведомость_разбивка.png")

fig = df2img.plot_dataframe(doh_k_zp_MC.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП МЦ (по ведомости) к Доходу МЦ (по FR) за {spravka_month.get(month_pf)} в разбивке, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 170),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_мц_к_зп_мц_ведомость_разбивка.png")

fig = df2img.plot_dataframe(doh_k_zp_KC.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП КЦ (по ведомости) к Доходу КЦ за {spravka_month.get(month_pf)} в разбивке, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 170),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_кц_к_зп_кц_ведомость_разбивка.png")

fig = df2img.plot_dataframe(doh_itog_k_zp_itog.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП всего автоцентра (по ведомости) к Доходу итого (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_итого_к_зп_итого_ведомость_с_н_г.png")

with open("process_progress.txt", "w") as file:  # заполняем прогрессбар
    file.write("60")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_itog_k_zp_itog.df['месяц'], doh_itog_k_zp_itog.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()

    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_itog_k_zp_itog.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_itog_k_zp_itog.name}.png')

fig = df2img.plot_dataframe(doh_itog_k_zp_ruk.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП РУК АС_ТЦ_АЦ (по ведомости) к Доходу итого (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_итого_к_зп_РУК_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_itog_k_zp_ruk.df['месяц'], doh_itog_k_zp_ruk.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()

    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_itog_k_zp_ruk.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_itog_k_zp_ruk.name}.png')


fig = df2img.plot_dataframe(doh_itog_k_zp_AC.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП АЦ.администрация без РУК АЦ (по ведомости) к Доходу итого (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_итого_к_зп_АЦ_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_itog_k_zp_AC.df['месяц'], doh_itog_k_zp_AC.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()

    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_itog_k_zp_AC.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_itog_k_zp_AC.name}.png')

fig = df2img.plot_dataframe(doh_TC_k_zp_TC.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП ТЦ без РУК ТЦ (по ведомости) к Доходу ТЦ (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_ТЦ_к_зп_ТЦ_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_TC_k_zp_TC.df['месяц'], doh_TC_k_zp_TC.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)
    plt.legend()
    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")
    plt.title(f"{doh_TC_k_zp_TC.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %", fontsize=9,
              fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_TC_k_zp_TC.name}.png')


fig = df2img.plot_dataframe(doh_AS_k_zp_AS.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП АС общая без РУК АС (по ведомости) к Доходу АС общему (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_АСобщ_к_зп_АСобщ_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_AS_k_zp_AS.df['месяц'], doh_AS_k_zp_AS.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()

    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_AS_k_zp_AS.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %", fontsize=9,
              fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_AS_k_zp_AS.name}.png')
    #plt.show()
fig = df2img.plot_dataframe(doh_ASnew_k_zp_ASnew.df.apply(infinity_).apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП АС новых ам (отдел_ОПА) без РУК АС (по ведомости) к Доходу АС новых ам (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_АСнов_к_зп_АСнов_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_ASnew_k_zp_ASnew.df['месяц'], doh_ASnew_k_zp_ASnew.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()
    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_ASnew_k_zp_ASnew.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_ASnew_k_zp_ASnew.name}.png')


fig = df2img.plot_dataframe(doh_AS_k_zp_AS_ovp.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП АС ОВП (отдел_ОВП) без РУК АС (по ведомости) к Доходу АС ОВП (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_АСовп_к_зп_АСовп_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_AS_k_zp_AS_ovp.df['месяц'], doh_AS_k_zp_AS_ovp.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()
    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_AS_k_zp_AS_ovp.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_AS_k_zp_AS_ovp.name}.png')


fig = df2img.plot_dataframe(doh_MC_k_zp_MC.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП ТЦ_МЦ (по ведомости) к Доходу ТЦ_МЦ (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_ТЦМЦ_к_зп_ТЦМЦ_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_MC_k_zp_MC.df['месяц'], doh_MC_k_zp_MC.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()
    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_MC_k_zp_MC.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %", fontsize=9,
              fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_MC_k_zp_MC.name}.png')


fig = df2img.plot_dataframe(doh_MC_k_zp_MC_meh.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП ТЦ_МЦ_механики (по ведомости) к Доходу ТЦ_МЦ (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_ТЦМЦ_к_зп_ТЦМЦмеханики_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_MC_k_zp_MC_meh.df['месяц'], doh_MC_k_zp_MC_meh.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()
    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_MC_k_zp_MC_meh.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_MC_k_zp_MC_meh.name}.png')


fig = df2img.plot_dataframe(doh_MC_k_zp_MC_itr.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП ТЦ_МЦ_итр (по ведомости) к Доходу ТЦ_МЦ (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_ТЦМЦ_к_зп_ТЦМЦитр_ведомость_с_н_г.png")


with open("process_progress.txt", "w") as file:  # заполняем прогрессбар
    file.write("80")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_MC_k_zp_MC_itr.df['месяц'], doh_MC_k_zp_MC_itr.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()
    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_MC_k_zp_MC_itr.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_MC_k_zp_MC_itr.name}.png')


fig = df2img.plot_dataframe(doh_KC_k_zp_KC.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП ТЦ_КЦ (по ведомости) к Доходу ТЦ_КЦ (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_ТЦКЦ_к_зп_ТЦКЦ_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_KC_k_zp_KC.df['месяц'], doh_KC_k_zp_KC.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()
    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_KC_k_zp_KC.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %", fontsize=9,
              fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_KC_k_zp_KC.name}.png')

fig = df2img.plot_dataframe(doh_KC_k_zp_KC_meh.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП ТЦ_КЦ_механики (по ведомости) к Доход ТЦ_КЦ (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_ТЦКЦ_к_зп_ТЦКЦмех_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_KC_k_zp_KC_meh.df['месяц'], doh_KC_k_zp_KC_meh.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()
    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_KC_k_zp_KC_meh.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_KC_k_zp_KC_meh.name}.png')

fig = df2img.plot_dataframe(doh_KC_k_zp_KC_itr.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП ТЦ_КЦ_итр (по ведомости) к Доходу ТЦ_КЦ (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_ТЦКЦ_к_зп_ТЦКЦитр_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_KC_k_zp_KC_itr.df['месяц'], doh_KC_k_zp_KC_itr.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()
    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_KC_k_zp_KC_itr.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_KC_k_zp_KC_itr.name}.png')


fig = df2img.plot_dataframe(doh_DO_k_zp_DO.df.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.3, 0.2, 0.2, 0.1],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"ЗП ТЦ_ДО (по ведомости) к Доход ТЦ_ДО (по FR), с начала года, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_доход_ТЦДО_к_зп_ТЦДО_ведомость_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(doh_DO_k_zp_DO.df['месяц'], doh_DO_k_zp_DO.df[f'ФЗП_от_ДОХ_%'], label=f'ФЗП_от_ДОХ_%',
             color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)

    plt.legend()
    plt.ylabel('Проценты', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"{doh_DO_k_zp_DO.name} \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf}г в %", fontsize=9,
              fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'{doh_DO_k_zp_DO.name}.png')

dfi.export(res_x_vr.fillna(0), 'df_результат_деятельности_по_выручке_с_н_г.png', dpi=200)
dfi.export(res_x_doh.fillna(0), 'df_результат_деятельности_по_доходу_с_н_г.png', dpi=200)

with plt.style.context('seaborn-v0_8'):  # применим стиль bmh / fivethirtyeight / seaborn-deep/seaborn-v0_8
    mmm = [i for i in df_PF[df_PF['месяц_digit'] <= month_pf]['месяц'].unique()]
    itogo_autom = [round(float(df_PF[(df_PF['индекс'] == ind_pf_revenue_all)
                                     & (df_PF['автоцентр'] == name_ac)
                                     & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(0).drop_duplicates(
        'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]
    MMM = [f"{i[0:3]}\n всего\n {round(y, 2)}" for i, y in zip(mmm, itogo_autom)]

    month_doh = MMM  # фильтруем df по месяцу отчета выбираем месяца
    doh_autocentr = {
        'Выручка_АС': np.array([round(float(df_PF[(df_PF['индекс'] == ind_pf_revenue_AS)
                                                  & (df_PF['автоцентр'] == name_ac)
                                                  & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(
            0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]),
        'Выручка_ТЦ': np.array([round(float(df_PF[(df_PF['индекс'] == ind_pf_revenue_mex_itg)
                                                  & (df_PF['автоцентр'] == name_ac)
                                                  & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(
            0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]),

    }

    x = np.arange(len(month_doh))  # the label month
    width = 0.15  # ширина полос
    multiplier = 0

    fig, ax = plt.subplots(figsize=(0.5 + (month_pf * 1.2), 10), dpi=200)

    for name_doh, doh_value in doh_autocentr.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, doh_value, width, label=name_doh)
        ax.bar_label(rects, rotation=90, padding=3, fontsize=16)
        multiplier += 1.3  # растояние между столбцами одной группы
    # скрипт определяющий построение графика по оси y min max для set_ylim
    mx = 0
    mi = 0
    for k, v in doh_autocentr.items():
        if max(v) > mx:
            mx = max(v)
        if min(v) < mi:
            mi = min(v)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('млн.руб', fontsize=16, fontweight="bold")
    ax.set_title(f'Распр-ие Выручки \n по {spravka_new_name.get(name_ac, name_ac)}\n в млн.р', fontweight="bold",
                 fontsize=13)
    ax.set_xticks(x + width, month_doh)
    ax.legend(loc='upper left', ncols=1)  # ncols=1 - столбцы легенды
    ax.set_ylim(mi - 3, mx + 15)  # + корреткировка +2 млн

    plt.xticks(rotation=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('месяц / Выручка', fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig('Выручка_.png')  # сохряняет в картику

df_revenue_ALL_res = func_IND_podr_year(df_PF, ind_pf_revenue_all, 'факт', 'Выручка')
df_revenue_ALL_res[f'Выручка_с накоплением_{year_pf}'] = df_revenue_ALL_res[f'факт_Выручка_{year_pf}'].cumsum()
df_revenue_ALL_res = convertor_zeros(df_revenue_ALL_res)

df_revenue_ALL_res_2 = func_IND_podr_year(df_PF, ind_pf_revenue_all, 'факт', 'Выручка', year=year_pf-1)
df_revenue_ALL_res_2[f'Выручка_с накоплением_{year_pf-1}'] = df_revenue_ALL_res_2[f'факт_Выручка_{year_pf-1}'].cumsum()
df_revenue_ALL_res_2 = convertor_zeros(df_revenue_ALL_res_2)

fig = df2img.plot_dataframe(df_revenue_ALL_res.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.2, 0.2, 0.2],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"Выручка с начала года и с накоплением, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_выручка_накопл_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep /seaborn-v0_8
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(df_revenue_ALL_res['месяц'],
             df_revenue_ALL_res[f'Выручка_с накоплением_{year_pf}'],
             label=f'Выручка_с_накоплением_{year_pf}', color='cornflowerblue')

    plt.plot(df_revenue_ALL_res_2['месяц'],
             df_revenue_ALL_res_2[f'Выручка_с накоплением_{year_pf - 1}'],
             label=f'Выручка_с_накоплением_{year_pf - 1}', color='green')

    plt.xticks(rotation=0)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.legend()
    plt.ylabel('Млн.руб', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"Выручка \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf - 1}-{year_pf}г в %", fontsize=9,
              fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'ВЫРУЧКА_с_накопл.png')

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep /seaborn-v0_8
    plt.figure(figsize=(12, 6), dpi=200)  # размер графика
    plt.plot(dynamic_2_years(df_PF, ind_pf_revenue_all)['Дата'],
             dynamic_2_years(df_PF, ind_pf_revenue_all)['факт'],
             label='Выручка_факт', color='cornflowerblue')

    # plt.plot(om_result_dinamic['Дата'], om_result_dinamic['план'], label='ОМ_план', color='red') # этустроку убрать

    # градус подписей по оси Х
    plt.xticks(rotation=0)
    # добавляем ось X
    plt.axhline(y=0, color='red', linestyle='--')

    plt.legend()
    plt.ylabel('Выручка', fontsize=10, fontweight="bold")
    plt.xlabel('ПЕРИОД', fontsize=10, fontweight="bold")
    plt.title(f"Выручка \n по {spravka_new_name.get(name_ac, name_ac)} \nза {year_pf - 1}-{year_pf}г в млн.руб",
              fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig('Выручка_динамика_2_года.png')  # сохряняет в картику

# БЛОК С РАСХОДАМИ


with plt.style.context('seaborn-v0_8'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    mmm = [i for i in df_PF[df_PF['месяц_digit'] <= month_pf]['месяц'].unique()]
    itogo_autom = [round(float(df_PF[(df_PF['индекс'] == ind_expenses_all)
                                     & (df_PF['автоцентр'] == name_ac)
                                     & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(0).drop_duplicates(
        'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]
    MMM = [f"{i[0:3]}\n всего\n {round(y, 2)}" for i, y in zip(mmm, itogo_autom)]

    month_doh = MMM  # фильтруем df по месяцу отчета выбираем месяца
    doh_autocentr = {
        'Расходы_АЦ': np.array([round(float(df_PF[(df_PF['индекс'] == ind_expenses_AC)
                                                  & (df_PF['автоцентр'] == name_ac)
                                                  & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(
            0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]),
        'Расходы_АС': np.array([round(float(df_PF[(df_PF['индекс'] == ind_expenses_AS)
                                                  & (df_PF['автоцентр'] == name_ac)
                                                  & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(
            0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]),
        'Расходы_ТЦ': np.array([round(float(df_PF[(df_PF['индекс'] == ind_expenses_TC)
                                                  & (df_PF['автоцентр'] == name_ac)
                                                  & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(
            0).drop_duplicates('месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]),

    }

    x = np.arange(len(month_doh))  # the label month
    width = 0.15  # ширина полос
    multiplier = 0

    fig, ax = plt.subplots(figsize=(0.5 + (month_pf * 1.2), 10), dpi=200)

    for name_doh, doh_value in doh_autocentr.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, doh_value, width, label=name_doh)
        ax.bar_label(rects, rotation=90, padding=3, fontsize=16)
        multiplier += 1.3  # растояние между столбцами одной группы
    # скрипт определяющий построение графика по оси y min max для set_ylim
    mx = 0
    mi = 0
    for k, v in doh_autocentr.items():
        if max(v) > mx:
            mx = max(v)
        if min(v) < mi:
            mi = min(v)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('млн.руб', fontsize=16, fontweight="bold")
    ax.set_title(f'Расходы \n по {spravka_new_name.get(name_ac, name_ac)}\n в млн.р', fontweight="bold", fontsize=13)
    ax.set_xticks(x + width, month_doh)
    ax.legend(loc='upper left', ncols=1)  # ncols=1 - столбцы легенды
    ax.set_ylim(mi - 1, mx + 3)  # + корреткировка +2 млн

    plt.xticks(rotation=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('месяц / Расходы', fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig('РАСХОДЫ_.png')  # сохряняет в картику

with open("process_progress.txt", "w") as file:  # для заполнения прогрессбара
    file.write('85')

df_rash_post = pd.merge(func_IND_podr_year(df_PF, ind = ind_expenses_all, fkt_or_pln = 'факт',ren='ВСЕГО'),
                        func_IND_podr_year(df_PF, ind = ind_expenses_ZP, fkt_or_pln = 'факт',ren='ЗП'))

df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_nalog_fot, fkt_or_pln = 'факт',ren='Налог_ФОТ'))
df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_rent, fkt_or_pln = 'факт',ren='Аренда'))
df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_advertising, fkt_or_pln = 'факт',ren='Реклама'))
df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_connection, fkt_or_pln = 'факт',ren='Связь'))
df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_pzv_njd, fkt_or_pln = 'факт',ren='Пр_нужды'))
df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_hoz_njd, fkt_or_pln = 'факт',ren='Хоз_нужды'))
df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_pc_po, fkt_or_pln = 'факт',ren='Комп_по'))
df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_proch, fkt_or_pln = 'факт',ren='Прочие'))
df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_ub, fkt_or_pln = 'факт',ren='Убытки'))
df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_amrt, fkt_or_pln = 'факт',ren='Амортизация'))
df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_amrt_dop, fkt_or_pln = 'факт',ren='Амортизация_доп'))
df_rash_post = pd.merge(df_rash_post,
                        func_IND_podr_year(df_PF, ind = ind_expenses_security, fkt_or_pln = 'факт',ren='Охрана'))

df_rash_post = df_rash_post.rename(columns={f'факт_ЗП_{year_pf}': 'ЗП', f'факт_Налог_ФОТ_{year_pf}': 'Нал_ФОТ',
                                            f'факт_Аренда_{year_pf}': 'Аренда', f'факт_Реклама_{year_pf}': 'Реклама',
                                            f'факт_Связь_{year_pf}': 'Связь', f'факт_Пр_нужды_{year_pf}': 'Пр_нужд',
                                            f'факт_Хоз_нужды_{year_pf}': 'Хоз_нужд', f'факт_Комп_по_{year_pf}': 'ПК_и_ПО',
                                            f'факт_Прочие_{year_pf}': 'Прочие', f'факт_Убытки_{year_pf}': 'Убытки',
                                            f'факт_Амортизация_{year_pf}': 'Амртз', f'факт_Охрана_{year_pf}': 'Охрана',
                                            f'факт_ВСЕГО_{year_pf}': 'Всего', f'факт_Амортизация_доп_{year_pf}': 'Амртз_доп'})

df_rash_post_percent = df_rash_post.copy()

df_rash_post = convertor_zeros(df_rash_post)
drop_colm_post_rsh = [i for i in df_rash_post_percent.columns if i not in ('месяц')]
for i in df_rash_post_percent.columns:
    if i not in ('месяц', 'Всего'):
        df_rash_post_percent[f'{i}'] = round((df_rash_post_percent[f'{i}']/df_rash_post_percent['Всего'])*100,2)
df_rash_post_percent.drop(['Всего'], axis=1, inplace=True)

fig = df2img.plot_dataframe(
      df_rash_post.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.2, 0.15, 0.18, 0.2, 0.18, 0.2, 0.15, 0.2, 0.2, 0.2, 0.18, 0.18, 0.15, 0.2, 0.15],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text="Распределение постоянных расходов (по FR), млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=14,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1200, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_распределение_постоянных_расходов_млн.png")

fig = df2img.plot_dataframe(
      df_rash_post_percent.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.2, 0.15, 0.18, 0.2, 0.18, 0.2, 0.15, 0.2, 0.2, 0.2, 0.18, 0.18, 0.15, 0.2, 0.15],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text="Распределение постоянных расходов (по FR), проценты %",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=14,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1200, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_распределение_постоянных_расходов_процент.png")

df_expenses_all_res = func_IND_podr_year(df_PF, ind_expenses_all, 'факт', 'РАСХОД')
df_expenses_all_res[f'РАСХОД_с накоплением_{year_pf}'] = df_expenses_all_res[f'факт_РАСХОД_{year_pf}'].cumsum()
df_expenses_all_res = convertor_zeros(df_expenses_all_res)

df_expenses_all_res_2 = func_IND_podr_year(df_PF, ind_expenses_all, 'факт', 'РАСХОД', year=year_pf-1)
df_expenses_all_res_2[f'РАСХОД_с накоплением_{year_pf-1}'] = df_expenses_all_res_2[f'факт_РАСХОД_{year_pf-1}'].cumsum()
df_expenses_all_res_2 = convertor_zeros(df_expenses_all_res_2)

fig = df2img.plot_dataframe(df_expenses_all_res.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.2, 0.2, 0.2],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"Расходы постоянные с начала года и с накоплением, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_расходы_постоянные_накопл_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(df_expenses_all_res['месяц'],
             df_expenses_all_res[f'РАСХОД_с накоплением_{year_pf}'],
             label=f'РАСХОД_с_накоплением_{year_pf}', color='cornflowerblue')

    plt.plot(df_expenses_all_res_2['месяц'],
             df_expenses_all_res_2[f'РАСХОД_с накоплением_{year_pf - 1}'],
             label=f'РАСХОД_с_накоплением_{year_pf - 1}', color='green')

    plt.xticks(rotation=0)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.legend()
    plt.ylabel('Млн.руб', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"Расходы \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf - 1}-{year_pf}г в %", fontsize=9,
              fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'РАСХОД_с_накопл.png')

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 6), dpi=200)  # размер графика
    plt.plot(dynamic_2_years(df_PF, ind_expenses_all)['Дата'],
             dynamic_2_years(df_PF, ind_expenses_all)['факт'],
             label='РАСХОДЫ_факт', color='cornflowerblue')

    # plt.plot(om_result_dinamic['Дата'], om_result_dinamic['план'], label='ОМ_план', color='red') # этустроку убрать

    # градус подписей по оси Х
    plt.xticks(rotation=0)
    # добавляем ось X
    # plt.axhline(y=0, color='red', linestyle='--')

    plt.legend()
    plt.ylabel('Расходы', fontsize=10, fontweight="bold")
    plt.xlabel('ПЕРИОД', fontsize=10, fontweight="bold")
    plt.title(
        f"Динамика Расходов \n по {spravka_new_name.get(name_ac, name_ac)} \nза {year_pf - 1}-{year_pf}г в млн.руб",
        fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig('РАСХОДЫ_динамика_2_года.png')  # сохряняет в картику

# БЛОК С БАЛАНСОВОЙ И ЧИСТОЙ


with plt.style.context('seaborn-v0_8'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    mmm = [i for i in df_PF[df_PF['месяц_digit'] <= month_pf]['месяц'].unique()]
    itogo_autom = [round(float(df_PF[(df_PF['индекс'] == ind_pf_balance_profit)
                                     & (df_PF['автоцентр'] == name_ac)
                                     & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(0).drop_duplicates(
        'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]
    MMM = [f"{i[0:3]}\n всего\n {round(y, 2)}" for i, y in zip(mmm, itogo_autom)]

    month_doh = MMM  # фильтруем df по месяцу отчета выбираем месяца
    doh_autocentr = {
        'БП_АС': np.array([round(float(df_PF[(df_PF['индекс'] == ind_pf_balance_profit_AS)
                                             & (df_PF['автоцентр'] == name_ac)
                                             & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(0).drop_duplicates(
            'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]),
        'БП_ТЦ': np.array([round(float(df_PF[(df_PF['индекс'] == ind_pf_balance_profit_TC)
                                             & (df_PF['автоцентр'] == name_ac)
                                             & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(0).drop_duplicates(
            'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]),

    }

    x = np.arange(len(month_doh))  # the label month
    width = 0.15  # ширина полос
    multiplier = 0

    fig, ax = plt.subplots(figsize=(0.5 + (month_pf * 1.2), 10), dpi=200)

    for name_doh, doh_value in doh_autocentr.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, doh_value, width, label=name_doh)
        ax.bar_label(rects, rotation=90, padding=3, fontsize=16)
        multiplier += 1.3  # растояние между столбцами одной группы
    # скрипт определяющий построение графика по оси y min max для set_ylim
    mx = 0
    mi = 0
    for k, v in doh_autocentr.items():
        if max(v) > mx:
            mx = max(v)
        if min(v) < mi:
            mi = min(v)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('млн.руб', fontsize=16, fontweight="bold")
    ax.set_title(f'Распред-ие БП \n по {spravka_new_name.get(name_ac, name_ac)} \n в млн.р', fontweight="bold",
                 fontsize=13)
    ax.set_xticks(x + width, month_doh)
    ax.legend(loc='upper left', ncols=1)  # ncols=1 - столбцы легенды
    ax.set_ylim(mi - 3, mx + 3)  # + корреткировка +2 млн

    plt.xticks(rotation=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('месяц / БП', fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig('БП_.png')  # сохряняет в картику

df_ballabce_profit_res = func_IND_podr_year(df_PF, ind_pf_balance_profit, 'факт', 'БП')
df_ballabce_profit_res[f'БП_с накоплением_{year_pf}'] = df_ballabce_profit_res[f'факт_БП_{year_pf}'].cumsum()
df_ballabce_profit_res = convertor_zeros(df_ballabce_profit_res)

df_ballabce_profit_res_2 = func_IND_podr_year(df_PF, ind_pf_balance_profit, 'факт', 'БП', year=year_pf-1)
df_ballabce_profit_res_2[f'БП_с накоплением_{year_pf-1}'] = df_ballabce_profit_res_2[f'факт_БП_{year_pf-1}'].cumsum()
df_ballabce_profit_res_2 = convertor_zeros(df_ballabce_profit_res_2)

fig = df2img.plot_dataframe(df_ballabce_profit_res.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.2, 0.2, 0.2],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"Балансовая прибыль с начала года и с накоплением, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_балансовая_прибыль_накопл_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(df_ballabce_profit_res['месяц'],
             df_ballabce_profit_res[f'БП_с накоплением_{year_pf}'],
             label=f'БП_с_накоплением_{year_pf}', color='cornflowerblue')

    plt.plot(df_ballabce_profit_res_2['месяц'],
             df_ballabce_profit_res_2[f'БП_с накоплением_{year_pf - 1}'],
             label=f'БП_с_накоплением_{year_pf - 1}', color='green')

    plt.xticks(rotation=0)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.legend()
    plt.ylabel('Млн.руб', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"Балансовая прибыль \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf - 1}-{year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'БП_с_накопл.png')

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 6), dpi=200)  # размер графика
    plt.plot(dynamic_2_years(df_PF, ind_pf_balance_profit)['Дата'],
             dynamic_2_years(df_PF, ind_pf_balance_profit)['факт'],
             label='БП_факт', color='cornflowerblue')

    # plt.plot(om_result_dinamic['Дата'], om_result_dinamic['план'], label='ОМ_план', color='red') # этустроку убрать

    # градус подписей по оси Х
    plt.xticks(rotation=0)
    # добавляем ось X
    plt.axhline(y=0, color='red', linestyle='--')

    plt.legend()
    plt.ylabel('Балансовая_прибыль', fontsize=10, fontweight="bold")
    plt.xlabel('ПЕРИОД', fontsize=10, fontweight="bold")
    plt.title(f"Динамика БП \n по {spravka_new_name.get(name_ac, name_ac)} \nза {year_pf - 1}-{year_pf}г в млн.руб",
              fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig('БП_динамика_2_года.png')  # сохряняет в картику

with plt.style.context('seaborn-v0_8'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    mmm = [i for i in df_PF[df_PF['месяц_digit'] <= month_pf]['месяц'].unique()]
    itogo_autom = [round(float(df_PF[(df_PF['индекс'] == ind_pf_net_profit)
                                     & (df_PF['автоцентр'] == name_ac)
                                     & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(0).drop_duplicates(
        'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]
    MMM = [f"{i[0:3]}\n всего\n {round(y, 2)}" for i, y in zip(mmm, itogo_autom)]

    month_doh = MMM  # фильтруем df по месяцу отчета выбираем месяца
    doh_autocentr = {
        'ЧП_АС': np.array([round(float(df_PF[(df_PF['индекс'] == ind_pf_net_profit_AS)
                                             & (df_PF['автоцентр'] == name_ac)
                                             & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(0).drop_duplicates(
            'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]),
        'ЧП_ТЦ': np.array([round(float(df_PF[(df_PF['индекс'] == ind_pf_net_profit_TC)
                                             & (df_PF['автоцентр'] == name_ac)
                                             & (df_PF['год'] == year_pf)][['факт', 'месяц']].fillna(0).drop_duplicates(
            'месяц').iloc[i, 0]) / 1000000, 2) for i in range(month_pf)]),

    }

    x = np.arange(len(month_doh))  # the label month
    width = 0.15  # ширина полос
    multiplier = 0

    fig, ax = plt.subplots(figsize=(0.5 + (month_pf * 1.2), 10), dpi=200)

    for name_doh, doh_value in doh_autocentr.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, doh_value, width, label=name_doh)
        ax.bar_label(rects, rotation=90, padding=3, fontsize=16)
        multiplier += 1.3  # растояние между столбцами одной группы
    # скрипт определяющий построение графика по оси y min max для set_ylim
    mx = 0
    mi = 0
    for k, v in doh_autocentr.items():
        if max(v) > mx:
            mx = max(v)
        if min(v) < mi:
            mi = min(v)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('млн.руб', fontsize=16, fontweight="bold")
    ax.set_title(f'Распред-ие ЧП \n по {spravka_new_name.get(name_ac, name_ac)}\n в млн.р', fontweight="bold",
                 fontsize=13)
    ax.set_xticks(x + width, month_doh)
    ax.legend(loc='upper left', ncols=1)  # ncols=1 - столбцы легенды
    ax.set_ylim(mi - 3, mx + 3)  # + корреткировка +2 млн

    plt.xticks(rotation=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('месяц / ЧП', fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig('ЧП_.png')  # сохряняет в картику

df_profit_res = func_IND_podr_year(df_PF, ind_pf_net_profit, 'факт', 'ЧП')
df_profit_res[f'ЧП_с накоплением_{year_pf}'] = df_profit_res[f'факт_ЧП_{year_pf}'].cumsum()
df_profit_res = convertor_zeros(df_profit_res)

df_profit_res_2 = func_IND_podr_year(df_PF, ind_pf_net_profit, 'факт', 'ЧП', year=year_pf-1)
df_profit_res_2[f'ЧП_с накоплением_{year_pf-1}'] = df_profit_res_2[f'факт_ЧП_{year_pf-1}'].cumsum()
df_profit_res_2 = convertor_zeros(df_profit_res_2)

fig = df2img.plot_dataframe(df_profit_res.apply(infinity_).fillna(0).astype(str),
      print_index=False, # отключает индексы
      col_width=[0.2, 0.2, 0.2],  # ширина колонок
      title=dict(
          font_color="black",
          font_family="Times New Roman",
          font_size=20,
          text=f"Чистая прибыль с начала года и с накоплением, млн.руб",
          x=0,
          xanchor="left",
      ),
      tbl_header=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=13,
        height=20,
        line_width=4,
    ),
      tbl_cells=dict(
        align="center",
        fill_color="light gray",
        font_color="black",
        font_size=16,
        height=30,
        line_width=3
    ),
      fig_size=(1100, 75+(30*month_pf)),
  )
df2img.save_dataframe(fig=fig, filename="df_чистая_прибыль_накопл_с_н_г.png")

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 5), dpi=200)  # размер графика
    plt.plot(df_profit_res['месяц'], df_profit_res[f'ЧП_с накоплением_{year_pf}'],
             label=f'ЧП_с_накоплением_{year_pf}', color='cornflowerblue')
    plt.plot(df_profit_res_2['месяц'], df_profit_res_2[f'ЧП_с накоплением_{year_pf - 1}'],
             label=f'ЧП_с_накоплением_{year_pf - 1}', color='green')

    # градус подписей по оси Х
    plt.xticks(rotation=0)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.legend()
    plt.ylabel('Млн.руб', fontsize=10, fontweight="bold")
    plt.xlabel('Период', fontsize=10, fontweight="bold")

    plt.title(f"Чистая прибыль \n по {spravka_new_name.get(name_ac, name_ac)} \n за {year_pf - 1}-{year_pf}г в %",
              fontsize=9, fontweight="bold")
    plt.tight_layout()  # вмещает все в сохраняемую картику
    plt.savefig(f'ЧП_с_накопл.png')

with plt.style.context('fivethirtyeight'):  # применим стиль bmh / fivethirtyeight / seaborn-deep
    plt.figure(figsize=(12, 6), dpi=200)  # размер графика
    plt.plot(dynamic_2_years(df_PF, ind_pf_net_profit)['Дата'],
             dynamic_2_years(df_PF, ind_pf_net_profit)['факт'],
             label='ЧП_факт', color='cornflowerblue')

    # градус подписей по оси Х
    plt.xticks(rotation=0)
    # добавляем ось X
    plt.axhline(y=0, color='red', linestyle='--')

    plt.legend()
    plt.ylabel('Чистая_прибыль', fontsize=10, fontweight="bold")
    plt.xlabel('ПЕРИОД', fontsize=10, fontweight="bold")
    plt.title(f"Динамика ЧП \n по {spravka_new_name.get(name_ac, name_ac)} \nза {year_pf - 1}-{year_pf}г в млн.руб",
              fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig('ЧП_динамика_2_года.png')  # сохряняет в картику


with open("process_progress.txt", "w") as file:  # для заполнения прогрессбара
    file.write('90')


pdf = FPDF()  # создаем объект класса
font_dir = font_dir  # путь к папке со шрифтами
pdf.add_page()  # создаем страницу
pdf.add_font(family="Serif", style="", fname=f"{font_dir}/DejaVuSansCondensed.ttf", uni=True)
pdf.set_font(family="Serif", size=12)

# оглавление титульник
pdf.cell(200, 10,
         txt=f'Анализ деятельности АЦ {spravka_new_name.get(name_ac, name_ac)} за {spravka_month.get(min(spravka_month.keys()))}-{spravka_month.get(month_pf)} {year_pf}г',
         align='C', ln=1)

pdf.set_font(family="Serif", size=8)
# pdf.ln()
pdf.image('df_распоряжение_оптимизации.png', h=25, w=180)

pdf.set_font(family="Serif", size=10)
# pdf.ln()
pdf.image('Динамика_дохода_1.png', h=100, w=18 * month_pf)

pdf.image('Динамика_дохода_2.png', h=100, w=18 * month_pf)
pdf.ln()
pdf.ln()
pdf.ln()
pdf.ln()

pdf.set_font(family="Serif", size=12)
pdf.cell(200, 10, txt=f'Динамика дохода с накоплением, в млн.руб', align='C')
pdf.ln()

pdf.image('df_дох_с_накопл.png', h=8 + (5 * df_last_first_target_x.count()[0]), w=190)  ########

pdf.image('ОМ_1.png', h=120, w=18 * month_pf)
pdf.image('ОМ_2.png', h=80, w=180)
pdf.image('ОМ_3.png', h=120, w=17 * month_pf)
pdf.image('df_распределение_дох_ом.png', h=20 + (5 * month_pf), w=180)
pdf.image('df_распределение_дох_ом_накопительно.png', h=20 + (5 * month_pf), w=180)

pdf.image('Распределение_ам.png', h=120, w=18 * month_pf)

pdf.image('df_доля_зп_к_доходу_итого.png', h=20 + (5 * month_pf), w=180)
pdf.image('Доля_ЗП_АЦ_1.png', h=80, w=180)  # если график линейный динамику не ставим

pdf.image('df_динамика_фзп_ас.png', h=20 + (5 * month_pf), w=180)
pdf.image('Динамика_ФЗП_АС.png', h=80, w=180)  # если график линейный динамику не ставим

pdf.image('df_динамика_фзп_тц.png', h=20 + (5 * month_pf), w=180)
pdf.image('Динамика_ФЗП_ТЦ.png', h=90, w=180)  # если график линейный динамику не ставим

pdf.image('df_доля_зп_подр_от_дох.png', h=20 + (5 * month_pf), w=180)
pdf.image('Доля_зп_подразделенй_от_дохода.png', h=90, w=180)  # если график линейный динамику не ставим

pdf.image('df_доход_к_зп_итого.png', h=20 + (5 * month_pf), w=180)
pdf.image('Доход_к_зп_итого_.png', h=90, w=180)  # если график линейный динамику не ставим
###########

pdf.image('df_лимит_зп_ац.png', h=8 + (5 * func_limit_ZP(df_ZP, lim_zp_max, ['АЦ']).count()[0]), w=190)
pdf.image('df_лимит_зп_ас.png', h=8 + (5 * func_limit_ZP(df_ZP, lim_zp_max, ['АС']).count()[0]), w=190)
pdf.image('df_лимит_зп_тц.png', h=8 + (5 * func_limit_ZP(df_ZP, lim_zp_max, ['ТЦ']).count()[0]), w=190)

pdf.image('df_доход_итого_к_зп_по_ведомости_итого.png', h=15 + (5 * doh_k_zp_itogo.count()[0]), w=180)
pdf.image('df_доход_ас_к_зп_ас_ведомость.png', h=15 + (5 * doh_k_zp_AS.count()[0]), w=180)
pdf.image('df_доход_тц_к_зп_тц_ведомость_разбивка.png', h=15 + (5 * doh_k_zp_TC.count()[0]), w=180)
pdf.image('df_доход_мц_к_зп_мц_ведомость_разбивка.png', h=15 + (5 * doh_k_zp_MC.count()[0]), w=180)
pdf.image('df_доход_кц_к_зп_кц_ведомость_разбивка.png', h=15 + (5 * doh_k_zp_KC.count()[0]), w=180)
pdf.image('df_доход_итого_к_зп_итого_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_itog_k_zp_itog.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_итого_к_зп_РУК_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_itog_k_zp_ruk.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_итого_к_зп_АЦ_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_itog_k_zp_AC.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_ТЦ_к_зп_ТЦ_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_TC_k_zp_TC.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_АСобщ_к_зп_АСобщ_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_AS_k_zp_AS.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_АСнов_к_зп_АСнов_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_ASnew_k_zp_ASnew.name}.png', h=80, w=180)  # если график линейный динамику не ставим

pdf.image('df_доход_АСовп_к_зп_АСовп_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_AS_k_zp_AS_ovp.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_ТЦМЦ_к_зп_ТЦМЦ_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_MC_k_zp_MC.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_ТЦМЦ_к_зп_ТЦМЦмеханики_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_MC_k_zp_MC_meh.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_ТЦМЦ_к_зп_ТЦМЦитр_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_MC_k_zp_MC_itr.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_ТЦКЦ_к_зп_ТЦКЦ_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_KC_k_zp_KC.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_ТЦКЦ_к_зп_ТЦКЦмех_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_KC_k_zp_KC_meh.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_ТЦКЦ_к_зп_ТЦКЦитр_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_KC_k_zp_KC_itr.name}.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('df_доход_ТЦДО_к_зп_ТЦДО_ведомость_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'{doh_DO_k_zp_DO.name}.png', h=80, w=180)  # если график линейный динамику не ставим

pdf.image('df_результат_деятельности_по_выручке_с_н_г.png', h=190, w=50 + (10 * month_pf))
pdf.image('df_результат_деятельности_по_доходу_с_н_г.png', h=190, w=50 + (10 * month_pf))

pdf.image('Выручка_.png', h=120, w=17 * month_pf)
pdf.image('df_выручка_накопл_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'ВЫРУЧКА_с_накопл.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('Выручка_динамика_2_года.png', h=80, w=180)

pdf.image('РАСХОДЫ_.png', h=120, w=17 * month_pf)
pdf.image('df_распределение_постоянных_расходов_млн.png', h=20 + (5 * month_pf), w=180)
pdf.image('df_распределение_постоянных_расходов_процент.png', h=20 + (5 * month_pf), w=180)
pdf.image('df_расходы_постоянные_накопл_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'РАСХОД_с_накопл.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('РАСХОДЫ_динамика_2_года.png', h=80, w=180)

pdf.image('БП_.png', h=120, w=17 * month_pf)
pdf.image('df_балансовая_прибыль_накопл_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'БП_с_накопл.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('БП_динамика_2_года.png', h=80, w=180)

pdf.image('ЧП_.png', h=120, w=17 * month_pf)
pdf.image('df_чистая_прибыль_накопл_с_н_г.png', h=20 + (5 * month_pf), w=180)
pdf.image(f'ЧП_с_накопл.png', h=80, w=180)  # если график линейный динамику не ставим
pdf.image('ЧП_динамика_2_года.png', h=80, w=180)

# БЛОК СОХРАНЕНИЯ ФАЙЛА

pdf.ln()
pdf.output(f"{save_string}/Анализ_АЦ_{spravka_new_name.get(name_ac, name_ac)}"
           f"_за_{spravka_month.get(min(spravka_month.keys()))}-"
           f"{spravka_month.get(month_pf)}_{year_pf}_г_"
           f"создан_{time_str_config}.pdf")


with open("process_progress.txt", "w") as file:  # для заполнения прогрессбара
    file.write('100')
