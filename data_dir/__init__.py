ID_TO_CLASS = {0:'N', 1:'AFIB', 2:'AFL', 3:'J'}
ID_TO_NAME = {0:'Normal',1:'atrial fibrillation',2:'atrial flutter',3:'AV junctional rhythm'}

AFDB_RECORDS = {'08378', '04746', '06426', '08405', '05091', '06995', '08455', '04908', '07910', '07162', '07879', '04015', '04048', '08434', '04936', '04043', '05261', '05121', '06453', '08215', '08219', '07859', '04126'}

CLASS_TO_ID = {}

for ID, beats in ID_TO_CLASS.items():
    for beat in beats:
        CLASS_TO_ID[beat] = ID
