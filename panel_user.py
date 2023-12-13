from tkinter import *
from tkinter import ttk
from tkinter.ttk import Progressbar 
import time 
from tkinter import Tk, Spinbox
from tkinter.messagebox import showerror, showwarning, showinfo
from tkinter import filedialog as fd
import subprocess
from my_func import *

root = Tk()
root.title("Анализ_данных_v.01")

# функция с отдельным вызовом окна
def clik_btn():
    # файл в который сохраняем переменные выбранные пользователем но сразу очищаем данные
    # для очистки файла в него запишем переменные которы считаем в отчете
    with open("glb_file.txt", "w") as file:
        file.write("")
    # для очистки файла прогресса
    with open("process_progress.txt","w") as file:
        file.write('0')

    print_item_values() # вызов функции подтверждения значений а далее отрабатывает визуал загрузки
    save_path() # вызов функции с полученным путем сохранения

    window = Tk()  
    window.title("Формирование_отчета")  
    window.geometry('300x25') 
    bar = Progressbar(window, length=300, orient=HORIZONTAL, value=100.0, phase=100)
    bar.pack(anchor=NW, padx=6, pady=6)

    try:
        subprocess.Popen(['python3', connect_fnc('run_analytics.py')]) # запуск скрипта анализа файла mac
    except:
        subprocess.Popen(['py', connect_fnc('run_analytics.py')])  # запуск скрипта анализа файла win
    # цикл считывания прогрессбара после запуска run_analytics.py по коду идет
    # запись прогресса в txt process_progress на увеличения числа от 10 до 100
    # градации по коду раскинул руками 10 30 60 80 100
    # а цикл ниже считывает эту запись
    x = 1
    while x < 100:
        with open("process_progress.txt", "r", encoding='utf-8') as file:
            x = int(file.readline())
            bar.grid(column=0, row=0)
            bar['value'] = x
            bar.update()
            time.sleep(1)
    window.destroy() # закрытие прогрссбара после отработки цикла
    print('Формирование_отчета')


def upd_bd():
    print('Нажата кнопка обновления БД')
    with open("process_progress.txt","w") as file:
        file.write('0')
    # запуск процесса обновления БД PF и прогрессбара
    window = Tk()
    window.title("Обновление_БД_PF")
    window.geometry('300x25')
    bar = Progressbar(window, length=300, orient=HORIZONTAL, value=100.0, phase=100)
    bar.pack(anchor=NW, padx=6, pady=6)
    try:
        subprocess.Popen(['python3', connect_fnc('collection_PF.py')])
    except:
        subprocess.Popen(['py', connect_fnc('collection_PF.py')])
    x = 1
    while x < 100:
        with open("process_progress.txt", "r", encoding='utf-8') as file:
            x = int(file.readline())
            bar.grid(column=0, row=0)
            bar['value'] = x
            bar.update()
            time.sleep(1)
    window.destroy() # закрытие окна прогрессбара

    # запуск процесса обновления БД ZP и прогрессбара
    with open("process_progress.txt","w") as file:
        file.write('0')
    window = Tk()
    window.title("Обновление_БД_ZP")
    window.geometry('300x25')
    bar = Progressbar(window, length=300, orient=HORIZONTAL, value=100.0, phase=100)
    bar.pack(anchor=NW, padx=6, pady=6)
    try:
        subprocess.Popen(['python3', connect_fnc('collection_ZP.py')])
    except:
        subprocess.Popen(['py', connect_fnc('collection_ZP.py')])
    x = 1
    while x < 100:
        with open("process_progress.txt", "r", encoding='utf-8') as file:
            x = int(file.readline())
            bar.grid(column=0, row=0)
            bar['value'] = x
            bar.update()
            time.sleep(1)
    window.destroy() # закрытие окна прогрессбара
def open_info(): 
    showinfo(title="Информация", message=mess)


mess = f"""
Автоцентр: 
- выбираем автоцентр  

Месяц: 
-выбираем месяц   

Год:
-выбираем год  

Обновить БД:
- использовать по необходимости, т.к. данные обновляются ежедневно в 11:10
- обновляет БАЗУ ДАННЫХ по ЗП и PF откуда будет формироваться отчет

Сформировать отчет:
- по нажатию бует открыто окно для выбора места сохранения отчета
- формирует отчет и выгружает его в формате PDF

По всем вопросам:
доб. 1377 Крутько Сергей
"""

item_1 = StringVar()
item_2 = StringVar()
item_3 = StringVar()

def print_item_values():
    """подвтерждает выбранные знаечния
    пока только выводит значение перемнных в строку и возвращает их, осталось привязать их к отчету
    передать значения переменным месяца и подразделения

    Returns:
        _type_: _description_
    """

    print(item_1.get(), item_2.get(), item_3.get())
    with open("glb_file.txt", "a") as file: # a дозапись автоцентра выбранного userom
        file.write(f'{item_1.get()}\n')
    with open("glb_file.txt", "a") as file: # a дозапись месяца выбранного userom
        file.write(f'{item_2.get()}\n')
    with open("glb_file.txt", "a") as file: # a дозапись года выбранного userom
        file.write(f'{item_3.get()}\n')


def save_path():
    """Функция выбора директории сохранения отчета
    передать путь сохранения отчета

    Returns:
        _type_: str - путь
    """
    root = Tk()
    root.withdraw()
    result = fd.askdirectory(
        master = root,
        mustexist=True
    )
    root.destroy()
    with open("glb_file.txt", "a") as file: # a дозапись сохраняет путь сохр.файла который выбрал user
        file.write(f'{repr(result)}\n')
    print(repr(result))
    #return repr(result)


Label(text="Автоцентр:").grid(row=1, column=0, sticky=W, padx=10, pady=10)
autocentre_label = ['yzs', 'yzm', 'cnt', 'yzk', 'sar', 'yti', 'yh', 'ychr', 'yr', 'yv']
item_1 = ttk.Combobox(root, values=autocentre_label, width=10)
item_1.grid(row=1, column=1, padx=10)

Label(text="Месяц:").grid(row=1, column=2, sticky=E)
months_label = ["Январь", "Февраль", "Март", 'Апрель', 'Май', 'Июнь',
                'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
item_2 = ttk.Combobox(root, values=months_label, width=10)
item_2.grid(row=1, column=3, sticky=E, padx=10)

Label(text="Год:").grid(row=1, column=4, sticky=W, padx=10, pady=10)
year_label = [2022, 2023, 2024]
item_3 = ttk.Combobox(root, values= year_label, width=5)
item_3.grid(row=1, column=4, padx=10)


Button(text="Справка", command=open_info).grid(row=2, column=0, pady=10, padx=10)
Button(text="Обновить БД", command=upd_bd).grid(row=2, column=1)
Button(text="Сформировать Отчет", command=clik_btn).grid(row=2, column=4, padx=10)


root.mainloop()

