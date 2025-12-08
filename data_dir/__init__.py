ID_TO_CLASS = {0:'N', 1:'AFIB', 2:'AFL', 3:'J'}
ID_TO_NAME = {0:'Normal',1:'atrial fibrillation',2:'atrial flutter',3:'AV junctional rhythm'}

AFDB_RECORDS = [
    "04015", "04043", "04048", "04126", "04746",
    "04908", "04936", "05091", "05121", "05261",
    "06426", "06453", "06995", "07162", "07859",
    "07879", "07910", "08215", "08219", "08378",
    "08405", "08434", "08455", "08479", "08541"
]

CLASS_TO_ID = {}

for ID, beats in ID_TO_CLASS.items():
    for beat in beats:
        CLASS_TO_ID[beat] = ID
