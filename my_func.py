import os

def connect_fnc(name_fnc:str):
    """
    Функция находит функции для вызова в сабпроцессах
    :param name_fnc: (str) - имя функции
    :return:
    """
    fnc = name_fnc
    lst_connect = ['//sim.local/data/Varsh/OFFICE/CAGROUP/run_python/task_scheduler/',
                   '/Users/sergey_krutko/PycharmProjects/analytics/']
    connect = ''
    counter = 0
    for i in lst_connect:
        if os.path.isdir(f"{i}"):
            print(f"Директория найдена {i}")
            if os.path.exists(f"{i}{fnc}"):
                print(f"Файл найден {i}{fnc}")
                connect = f"{i}{fnc}"
                counter+=1
                break
        else:
            print(f"Директории нет {i}")
            print(f'Ищем дальше')

    if counter != 0:
        print(f'Функция подключена к директории {connect}')
    return connect


