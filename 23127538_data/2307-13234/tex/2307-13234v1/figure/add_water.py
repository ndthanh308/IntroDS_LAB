import os
import glob

for inf in glob.glob('*.pdf'):
    print(inf)
    outf = inf.replace('.pdf', '_addwatermark.pdf')
    print(outf)
    os.system(f'pdftk {inf} stamp fig_watermark.pdf output {outf}')
    # svg = outf.replace('.pdf', '.svg')
    # svgf = f'../svg/{svg}'
    # os.system(f'pdf2svg {outf} {svgf}')