import os

import numpy as np
import pandas as pd
import re

# df = pd.read_csv('drive/My Drive/P.csv', sep=';', decimal=',', dtype={'PatientKey': 'Int32'}, encoding='utf-8',
#                  parse_dates=['BirthDate', 'LaboratoryResultsDate', 'MinLaboratoryResultsDate'])


def Gleb(df):
    def lab_result_proc(s):
        '''
        The best method for describing RegEx:
        r"""(?x)\d +  # the integral part
                \.    # the decimal point
                \d *  # some fractional digits"""
        '''
        pattern_1 = r'^\^?\s?[<>]?=?\s?(-?\d+[.,]?\d*).*$'
        pattern_2 = r'^(-?\d+[.,]?\d*)\s?-\s?(-?\d+[.,]?\d*)$'
        pattern_3 = r'.*\n?.*>=?\s?(-?\d+[.,]?\d*).*\n?.*'
        pattern_4 = r'.*\n?.*<=?\s?(-?\d+[.,]?\d*).*\n?.*'
        res_val = s['ValueText']
        res_num = re.fullmatch(pattern_1, res_val) if isinstance(res_val, str) else False
        res_val = float(res_num.group(1).replace(',', '.')) if res_num else np.nan
        ref_min = s['RefMin']
        ref_max = s['RefMax']
        ref_text = s['RefText']
        ref_range = re.fullmatch(pattern_2, ref_text) if isinstance(ref_text, str) else False
        ref_great = re.fullmatch(pattern_3, ref_text) if isinstance(ref_text, str) else False
        ref_less = re.fullmatch(pattern_4, ref_text) if isinstance(ref_text, str) else False
        ref_min = ref_min if np.isnan(ref_min) == False \
            else float(ref_range.group(1).replace(',', '.')) if ref_range \
            else float(ref_great.group(1).replace(',', '.')) if ref_great \
            else ref_min
        ref_max = ref_max if np.isnan(ref_max) == False \
            else float(ref_range.group(2).replace(',', '.')) if ref_range \
            else float(ref_less.group(1).replace(',', '.')) if ref_less \
            else ref_max
        return res_val, ref_min, ref_max

    def lab_result_value(s):
        result, _, _ = lab_result_proc(s)
        return result

    def lab_result_min(s):
        _, result, _ = lab_result_proc(s)
        return result

    def lab_result_max(s):
        _, _, result = lab_result_proc(s)
        return result

    def path_lab_result_proc(s):
        pattern_neg = r'(?is).*не обнар.*|.*отриц.*|.*отсут.*|.*не выявл.*|.*не найден.*|.*neg.*|.*norm.*|.*none.*|.*not found.*|^-*$'
        pattern_pos = r'(?is).*обнар.*|.*полож.*|.*присут.*|.*выявл.*|.*pos.*|^\+*$'
        res_val = s['Value']
        ref_min = s['RefMin']
        ref_max = s['RefMax']
        result = 0
        if pd.notnull(res_val):
            if isinstance(res_val, str):
                if re.fullmatch(pattern_neg, res_val):
                    result = 1
                elif re.fullmatch(pattern_pos, res_val):
                    result = 2
            elif pd.isnull(ref_min) and pd.notnull(ref_max):
                if (res_val <= ref_max):
                    result = 1
                else:
                    result = 2
            elif pd.notnull(ref_min) and pd.isnull(ref_max):
                if (ref_min <= res_val):
                    result = 1
                else:
                    result = 2
            elif pd.notnull(ref_min) and pd.notnull(ref_max) and isinstance(ref_min, float) and isinstance(ref_max,
                                                                                                           float):
                if (ref_min <= res_val <= ref_max):
                    result = 1
                else:
                    result = 2
        return result

    # 'Есть патология' if result == 2 else 'Нет патологии' if result == 1 else 'Не определенно'

    def age_group(elem):
        if elem['Age'] < 1:
            return 1
        elif 1 <= elem['Age'] <= 3:
            return 2
        elif 4 <= elem['Age'] <= 12:
            return 3
        elif 13 <= elem['Age'] < 18:
            return 4
        elif 18 <= elem['Age'] <= 29:
            return 5
        elif 30 <= elem['Age'] <= 45:
            return 6
        elif 46 <= elem['Age'] <= 59 and elem['Gender'] == 1:
            return 7
        elif 46 <= elem['Age'] <= 64 and elem['Gender'] == 0:
            return 8
        elif 60 <= elem['Age'] and elem['Gender'] == 1:
            return 9
        elif 65 <= elem['Age'] and elem['Gender'] == 0:
            return 10

    def path_lab_result_proc_2(s):
        pattern_neg = r'''(?is).*не обнар.*|.*отриц.*|.*отсут.*|.*не выявл.*|.*не найден.*|.*neg.*|.*norm.*|.*none.*|.*not found.*|
                                .*небольшое количество.*|.*незначительно.*|.*единичные в препарате.*|
                                .*полная.*|.*прозрачная.*|.*желтый.*|.*оттенки желтого.*|^-*$'''
        pattern_pos = r'(?is).*обнар.*|.*полож.*|.*присут.*|.*выявл.*|.*pos.*|^\+*$'
        res_val = s['ValueText']
        result = 0
        if pd.notnull(res_val):
            if isinstance(res_val, str):
                if re.fullmatch(pattern_neg, res_val):
                    result = 1
                elif re.fullmatch(pattern_pos, res_val):
                    result = 2
        return result

    filter3 = (df['Value'].isnull())
    df.loc[filter3, 'Value'] = df.loc[filter3].apply(lab_result_value, axis=1)
    filter4 = (df['RefMin'].isnull())
    df.loc[filter4, 'RefMin'] = df.loc[filter4].apply(lab_result_min, axis=1)
    filter5 = (df['RefMax'].isnull())
    df.loc[filter5, 'RefMax'] = df.loc[filter5].apply(lab_result_max, axis=1)
    df['Pathology'] = df.apply(path_lab_result_proc, axis=1)
    filter3 = (df['Value'].isnull())
    df.loc[filter3, 'Pathology'] = df.loc[filter3].apply(path_lab_result_proc_2, axis=1)
    filter3 = (df['Value'].isnull())
    df['Pathology'] = df['Pathology'] - 1
    df.loc[filter3, 'Value'] = df.loc[filter3, 'Pathology']
    df['Value'] = df.apply(lambda row: 0 if row['ValueText'] == row['RefText'] else row['Value'], axis=1)
    df = df[(df.Value <= 5000) | (df.Value.isnull())]
    data = df.pivot_table(index=['PatientKey']
                          , columns=['LaboratoryMethodsKey']
                          , values=['Value', 'Pathology']
                          , aggfunc='first'
                          , fill_value=-1
                          )

    data.reset_index(col_level=1).to_csv('lena.csv', encoding='utf-8', index=False)
    df_1 = pd.read_csv('lena.csv', sep=',', encoding='utf-8', header=1)
    a = ["25237", "25238", "25239", "25240", "25244", "25245", "25247", "25255",
         "25256", "25257", "25258", "25259", "25532", "25533", "25541", "26926", "9995650", "9995655",
         "9995658", "9995659", "9995660", "9995788", "9998728", "9998729", "9998730", "9998731",
         "9998732", "9998733", "9998735", "9998736", "9998737", "9998738", "9998739", "9998740",
         "9998741", "9998742", "9998744", "9998745", "9998746", "9998747", "9998748", "9998749",
         "9998750", "9998751", "9998838", "9998839", "9998840", "9998841", "9998842", "9998843",
         "9998844", "9998845", "9998846", "9998847", "9998848", "9998849", "9998850", "9998851",
         "9998854", "9998855", "9998856", "9998956", "9998957", "9998958", "9998959", "9998960",
         "9998961", "9998962", "9998963", "9998964", "9998965", "9998966", "9998967", "9998968",
         "9998969", "9998970", "9998971", "10002283", "10002284", "10002326", "10003007", "10005140",
         "10005142", "10005143", "10005144", "10005145", "10005157", "10005241", "10005298", "10005618",
         "10005629", "10006126", "10006304", "10006370", "10006436", "10006698", "10006816", "10006826",
         "10006848", "10007382", "10007384", "10007618", "10007619", "10007634", "10007635", "10007706",
         "10007707", "10007712", "10007716", "10007728", "10007733", "10007766", "10007779", "10007783",
         "10007784", "10007785", "10007786", "10007832", "10007833", "10007834", "10007884", "10007998",
         "10008000", "10008028", "10008163", "10008164", "10008196", "10009069", "10009070", "10009074",
         "10009406", "10009442", "10009483", "10009739", "10009740", "10010345", "10010641", "10010642",
         "10010643", "10010644", "10010645", "10010646", "10010647", "10010648", "10010649", "10010650",
         "10010651", "10010652", "10010653", "10010765", "10010766", "10010767", "10010787", "10010833",
         "10010837", "10010839", "10010841", "10010906", "10011290", "10011485", "10011487", "10011519",
         "10011520", "10011652", "10012970", "10013102", "10013103", "10013110", "10013111", "10013114",
         "10013115", "10013116", "10013117", "10013118", "10013152", "10013154", "10013157", "10014014",
         "10014056", "10014646", "10014648", "10014649", "10014713", "10014714", "10014715", "10014716",
         "10014717", "10014718", "10014719", "10014720", "10014721", "10014722", "10014723", "10014724",
         "10014725", "10014726", "10014912", "10014913", "10014914", "10014915", "10014916", "10014917",
         "10014918", "10014919", "10014920", "10014921", "10014922", "10014923", "10014924", "10014925",
         "10014926", "10014927", "10014928", "10014929", "10014930", "10014931", "10014932", "10014933",
         "10014934", "10014935", "10014936", "10014937", "10014938", "10014943", "10014944", "10014945",
         "10014946", "10014947", "10014948", "10014949", "10014950", "10014951", "10014952", "10014953",
         "10014954", "10014955", "10014956", "10014999", "10015017", "10015018", "10015019", "10015020",
         "10015021", "10015022", "10015023", "10015024", "10015025", "10015026", "10015027", "10015028",
         "10015029", "10015030", "10015031", "10015032", "10015033", "10015069", "10015070", "10015136",
         "10015146", "10015148", "10015149", "10015150", "10015151", "10015258", "10015259", "25237.1",
         "25238.1", "25239.1", "25240.1", "25244.1", "25245.1", "25247.1", "25255.1", "25256.1",
         "25257.1", "25258.1", "25259.1", "25532.1", "25533.1", "25541.1", "26926.1", "9995650.1",
         "9995655.1", "9995658.1", "9995659.1", "9995660.1", "9995788.1", "9998728.1", "9998729.1",
         "9998730.1", "9998731.1", "9998732.1", "9998733.1", "9998735.1", "9998736.1", "9998737.1",
         "9998738.1", "9998739.1", "9998740.1", "9998741.1", "9998742.1", "9998744.1", "9998745.1",
         "9998746.1", "9998747.1", "9998748.1", "9998749.1", "9998750.1", "9998751.1", "9998838.1",
         "9998839.1", "9998840.1", "9998841.1", "9998842.1", "9998843.1", "9998844.1", "9998845.1",
         "9998846.1", "9998847.1", "9998848.1", "9998849.1", "9998850.1", "9998851.1", "9998854.1",
         "9998855.1", "9998856.1", "9998956.1", "9998957.1", "9998958.1", "9998959.1", "9998960.1",
         "9998961.1", "9998962.1", "9998963.1", "9998964.1", "9998965.1", "9998966.1", "9998967.1",
         "9998968.1", "9998969.1", "9998970.1", "9998971.1", "10002283.1", "10002284.1", "10002326.1",
         "10003007.1", "10005140.1", "10005142.1", "10005143.1", "10005144.1", "10005145.1",
         "10005157.1", "10005241.1", "10005298.1", "10005618.1", "10005629.1", "10006126.1",
         "10006304.1", "10006370.1", "10006436.1", "10006698.1", "10006816.1", "10006826.1",
         "10006848.1", "10007382.1", "10007384.1", "10007618.1", "10007619.1", "10007634.1",
         "10007635.1", "10007706.1", "10007707.1", "10007712.1", "10007716.1", "10007728.1",
         "10007733.1", "10007766.1", "10007779.1", "10007783.1", "10007784.1", "10007785.1",
         "10007786.1", "10007832.1", "10007833.1", "10007834.1", "10007884.1", "10007998.1",
         "10008000.1", "10008028.1", "10008163.1", "10008164.1", "10008196.1", "10009069.1",
         "10009070.1", "10009074.1", "10009406.1", "10009442.1", "10009483.1", "10009739.1",
         "10009740.1", "10010345.1", "10010641.1", "10010642.1", "10010643.1", "10010644.1",
         "10010645.1", "10010646.1", "10010647.1", "10010648.1", "10010649.1", "10010650.1",
         "10010651.1", "10010652.1", "10010653.1", "10010765.1", "10010766.1", "10010767.1",
         "10010787.1", "10010833.1", "10010837.1", "10010839.1", "10010841.1", "10010906.1",
         "10011290.1", "10011485.1", "10011487.1", "10011519.1", "10011520.1", "10011652.1",
         "10012970.1", "10013102.1", "10013103.1", "10013110.1", "10013111.1", "10013114.1",
         "10013115.1", "10013116.1", "10013117.1", "10013118.1", "10013152.1", "10013154.1",
         "10013157.1", "10014014.1", "10014056.1", "10014646.1", "10014648.1", "10014649.1",
         "10014713.1", "10014714.1", "10014715.1", "10014716.1", "10014717.1", "10014718.1",
         "10014719.1", "10014720.1", "10014721.1", "10014722.1", "10014723.1", "10014724.1",
         "10014725.1", "10014726.1", "10014912.1", "10014913.1", "10014914.1", "10014915.1",
         "10014916.1", "10014917.1", "10014918.1", "10014919.1", "10014920.1", "10014921.1",
         "10014922.1", "10014923.1", "10014924.1", "10014925.1", "10014926.1", "10014927.1",
         "10014928.1", "10014929.1", "10014930.1", "10014931.1", "10014932.1", "10014933.1",
         "10014934.1", "10014935.1", "10014936.1", "10014937.1", "10014938.1", "10014943.1",
         "10014944.1", "10014945.1", "10014946.1", "10014947.1", "10014948.1", "10014949.1",
         "10014950.1", "10014951.1", "10014952.1", "10014953.1", "10014954.1", "10014955.1",
         "10014956.1", "10014999.1", "10015017.1", "10015018.1", "10015019.1", "10015020.1",
         "10015021.1", "10015022.1", "10015023.1", "10015024.1", "10015025.1", "10015026.1",
         "10015027.1", "10015028.1", "10015029.1", "10015030.1", "10015031.1", "10015032.1",
         "10015033.1", "10015069.1", "10015070.1", "10015136.1", "10015146.1", "10015148.1",
         "10015149.1", "10015150.1", "10015151.1", "10015258.1", "10015259.1", 'Gender', 'Age_days', 'Age', 'Age_group']

    for i in a:
        if i not in df_1:
            df_1[i] = -1

    df_1['Gender'] = df['Gender'].replace({'мужской': 1, 'женский': 0})
    df_1['Age_days'] = (np.datetime64('2021-12-31') - df['BirthDate']).dt.days
    df_1['Age'] = ((np.datetime64('2021-12-31') - df['BirthDate']).dt.days / 365.25).astype(int)
    df_1['Age_group'] = df_1.apply(age_group, axis=1)

    patalog = []

    for i in df_1:
        if list(df_1[i]) == [1] and i.isdigit():
            diag = df[df.LaboratoryMethodsKey == int(i)]['LaboratoryMethodsName']
            lst = ''.join(set(diag))
            patalog.append(lst)
    os.remove('lena.csv')

    return df_1.drop('PatientKey', axis=1), patalog
