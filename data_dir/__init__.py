ID_TO_BEAT = {0:['N','L','R','e','j'], 1:['S','A','a','J'], 2:['V','E'], 3:['F'], 4:['/','Q','f']}
ID_TO_AAMI = {0:'Normal', 1:'Supraventricular Premature', 2:'Premature Ventricular Contraction', 3:'Fusion of Ventricular & Normal', 4:'Unclassifiable'}
ID_TO_CLASS = {0:'N',1:'S',2:'V',3:'F',4:'Q'}

# 102, 104, 107, and 217
PATIENT_IDS = [100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]

BEAT_TO_ID = {}

for ID, beats in ID_TO_BEAT.items():
    for beat in beats:
        BEAT_TO_ID[beat] = ID
