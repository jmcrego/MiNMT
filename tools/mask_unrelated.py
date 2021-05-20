# -*- coding: utf-8 -*-
import sys
import edit_distance


class mask_unrelated():

    def __init__(self, u='✖', lc=False, o='dab'):
        self.u = u
        self.lc = lc

    def __call__(self, a, b):
        l1 = a.strip().split(' ')
        l2 = b.strip().split(' ')
        if len(l1) == 0 or l1[0] == '':
            l1 = []
        if len(l2) == 0 or l2[0] == '':
            l2 = []

        ratio = 0.0
        L1 = [self.u] * len(l1)
        L2 = [self.u] * len(l2)
        if len(l1) and len(l2):
            ### use .lower() or .casefold()
            sm = edit_distance.SequenceMatcher(a=[s.casefold() if self.lc else s for s in l1], b=[s.casefold() if self.lc else s for s in l2], action_function=edit_distance.highest_match_action)
            ratio = sm.ratio()
            ### initially all discarded
            for (code, b1, e1, b2, e2) in sm.get_opcodes():
                if code == 'equal': ### keep words
                    L1[b1] = l1[b1]
                    L2[b2] = l2[b2]

        out = []
        for c in o:
            if c == 'd':
                out.append('{:.6f}'.format(ratio))
            if c == 'a':
                out.append(' '.join(L1))
            if c == 'b':
                out.append(' '.join(L2))
        print('\t'.join(out))


if __name__ == '__main__':
    fa = None
    fb = None
    a = None
    b = None
    u = '✖'
    lc = False
    o = 'b'
    prog = sys.argv.pop(0)
    usage = '''usage: {} [-fa FILE -fb FILE] [-a STRING -b STRING] [-o STRING] [-u STRING] [-lc]
    -fa  FILE : a parallel file to compute unrelated words sentence-by-sentence
    -fb  FILE : b parallel file to compute unrelated words sentence-by-sentence
    -a STRING : a sentences to compute unrelated words
    -b STRING : b sentences to compute unrelated words
    -o STRING : output d:distance, a:first sentence b:second sentence (default {})
    -u STRING : token to mark unrelated words (default {})
    -lc       : lowercase string before computing edit distance (default {})
    -h        : this help
Needs edit_distance module: pip install edit_distance
'''.format(prog,o,u,lc)
    
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if tok=="-h":
            sys.stderr.write(usage);
            sys.exit()
        elif tok=="-fa":
            fa = sys.argv.pop(0)
        elif tok=="-fb":
            fb = sys.argv.pop(0)
        elif tok=="-a":
            a = sys.argv.pop(0)
        elif tok=="-b":
            b = sys.argv.pop(0)
        elif tok=="-lc":
            lc = True
        elif tok=="-u":
            lc = sys.argv.pop(0)
        elif tok=="-o":
            o = sys.argv.pop(0)
        else:
            sys.stderr.write('Unrecognized {} option\n'.format(tok))
            sys.stderr.write(usage)
            sys.exit()

    m = mask_unrelated(u=u, lc=lc, o=o)

    if a is not None and b is not None:
        m(a,b)

    if fa is not None and fb is not None:
        with open(fa) as f1, open(fb) as f2:
            for a, b in zip(f1, f2):
                m(a,b)
