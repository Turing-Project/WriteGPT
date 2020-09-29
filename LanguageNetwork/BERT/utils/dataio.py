#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: dataio.py
@time: 8/27/2019 4:31 PM
@desc:
"""

import os


def load_txt_data(path, mode='utf-8-sig', origin=False):
    """
    This func is used to reading txt file
    :param origin:
    :param path: path where file stored
    :param mode:
    :type path: str
    :return: string lines in file in a list
    :rtype: list
    """
    if type(path) != str:
        raise TypeError
    res = []

    file = open(path, 'rb')
    lines = file.read().decode(mode, 'ignore')
    for line in lines.split('\n'):
        line = line.strip()
        if origin:
            res.append(line)
        else:
            if line:
                res.append(line)
    file.close()
    return res


def load_excel_data(path):
    """
    This func is used to reading excel file
    :param path: path where file stored
    :type path: str
    :return: data saved in a pandas DataFrame
    :rtype: pandas.DataFrame
    """
    if type(path) != str:
        raise TypeError
    import pandas as pd
    return pd.read_excel(path).loc[:]


def load_variable(path):
    """
    :param path:
    :return:
    """
    import pickle
    return pickle.load(open(path, 'rb'))


def save_txt_file(data, path, end='\n'):
    """
    This func is used to saving data to txt file
    support data type:
    list: Fully support
    dict: Only save dict key
    str: will save single char to each line
    tuple: Fully support
    set: Fully support
    :param data: data
    :param path: path to save
    :type path: str
    :param end:
    :type end: str
    :return: None
    """
    if type(data) not in [list, dict, str, tuple, set] or type(path) != str:
        raise TypeError

    remove_old_file(path)

    with open(path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(str(item) + end)


def save_variable(variable, path):
    """
    :param variable:
    :param path:
    :return:
    """
    import pickle
    return pickle.dump(variable, open(path, 'wb'))


def load_file_name(path):
    """
    This func can get root, subdir, file_names
    :param path:
    :type path:str
    :return:
    """
    for root, dirs, files in os.walk(path):
        return root, dirs, files


def load_all_file_name(path, list_name, suffix='', not_include='.py'):
    """
    Load all file name including sub folder
    :param path:
    :param list_name:
    :param suffix:
    :param not_include:
    :return:
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path) and not_include not in file_path:
            load_all_file_name(file_path, list_name, suffix, not_include)
        elif os.path.splitext(file_path)[1] == suffix:
            list_name.append(file_path)


def check_dir(path):
    """
    check dir exists
    :param path:
    :type path:str
    :return:
    :rtype: bool
    """
    return os.path.exists(path)


def mkdir(path):
    """
    :param path:
    :type path: str
    :return: None
    """
    path = path.strip()
    if not check_dir(path):
        os.makedirs(path)


def remove_old_file(path):
    """
    :param path:
    :type path: str
    :return:
    """
    if check_dir(path):
        os.remove(path)


def delete_file(path):
    os.remove(path)


if __name__ == '__main__':
    pass
