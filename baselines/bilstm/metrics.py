import os
import stat
import subprocess
from os import chmod
from os.path import isfile


def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)


def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = 'conlleval.pl'

    if not isfile(_conlleval):
        # download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/conlleval.pl')
        os.system('wget https://www.comp.nus.edu.sg/%7Ekanmy/courses/practicalNLP_2008/packages/conlleval.pl')
        chmod('conlleval.pl', stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    # out = ['accuracy:', '16.26%;', 'precision:', '0.00%;', 'recall:', '0.00%;', 'FB1:', '0.00']

    precision = float(out[3][:-2])
    recall = float(out[5][:-2])
    f1score = float(out[7])

    return {'p': precision, 'r': recall, 'f1': f1score}
