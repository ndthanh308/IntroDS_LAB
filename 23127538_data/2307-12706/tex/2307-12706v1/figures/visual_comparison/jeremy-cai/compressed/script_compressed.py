import subprocess

IN_FILES = [
    'jeremy-cai-1174.png',
    'rateindex0-coolchic-R_0.801029-P_41.887096.png',
    'rateindex0-hevc-R_0.813128302-P_40.68264895.png',
    'rateindex13-coolchic-R_0.077039-P_29.55846.png',
    'rateindex13-hevc-R_0.073485978-P_27.94882861.png'
]

for in_f in IN_FILES:
    out_f = in_f[:-4] + '.jpeg'
    cmd = f'convert {in_f} -quality 94 -resize 85% {out_f}'
    subprocess.call(cmd, shell=True)