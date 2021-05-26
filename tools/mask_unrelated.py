# -*- coding: utf-8 -*-
import sys
import edit_distance


class mask_unrelated():

    def __init__(self, u='✖', d=0.6, lc=False, o='dab'):
        self.u = u
        self.lc = lc
        self.n_sents = 0
        self.n_unr_a = 0
        self.n_unr_b = 0
        self.n_tok_a = 0
        self.n_tok_b = 0

    def __call__(self, l1, l2):
        ratio = 0.0
        ### initially all unrelated
        L1 = [self.u] * len(l1)
        L2 = [self.u] * len(l2)
        self.n_sents += 1
        self.n_tok_a += len(L1)
        self.n_tok_b += len(L2)
        self.n_unr_a += len(L1)
        self.n_unr_b += len(L2)

        if len(l1) and len(l2):
            ### use .lower() or .casefold()
            sm = edit_distance.SequenceMatcher(a=[s.casefold() if self.lc else s for s in l1], b=[s.casefold() if self.lc else s for s in l2], action_function=edit_distance.highest_match_action)
            ratio = sm.ratio()
            if ratio >= d:
                for (code, b1, e1, b2, e2) in sm.get_opcodes():
                    if code == 'equal': ### related words
                        L1[b1] = l1[b1]
                        L2[b2] = l2[b2]
                        self.n_unr_a -= 1
                        self.n_unr_b -= 1

        out = []
        for c in o:
            if c == 'd':
                out.append('{:.6f}'.format(ratio))
            if c == 'a':
                out.append(' '.join(L1))
            if c == 'b':
                out.append(' '.join(L2))

        print('\t'.join(out))

    def stats(self):
        return self.n_sents, self.n_tok_a, self.n_tok_b, self.n_unr_a, self.n_unr_b


if __name__ == '__main__':
    fa = None
    fb = None
    a = None
    b = None
    d = 0.6
    u = '✖'
    lc = False
    o = 'b'
    prog = sys.argv.pop(0)
    usage = '''usage: {} [-fa FILE -fb FILE] [-a STRING -b STRING] [-d FLOAT] [-o STRING] [-u STRING] [-lc]
    -fa  FILE : a parallel file to compute unrelated words sentence-by-sentence
    -fb  FILE : b parallel file to compute unrelated words sentence-by-sentence
    -a STRING : a sentence to compute unrelated words
    -b STRING : b sentence to compute unrelated words
    -d  FLOAT : minimum distance (default {})
    -o STRING : output d:distance, a:first sentence b:second sentence (default {})
    -u STRING : token used to mark unrelated word (default {})
    -lc       : lowercase string before computing edit distance (default {})
    -h        : this help
Needs edit_distance module: pip install edit_distance
'''.format(prog,d,o,u,lc)
    
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
        elif tok=="-d":
            d = float(sys.argv.pop(0))
        elif tok=="-lc":
            lc = True
        elif tok=="-u":
            u = sys.argv.pop(0)
        elif tok=="-o":
            o = sys.argv.pop(0)
        else:
            sys.stderr.write('Unrecognized {} option\n'.format(tok))
            sys.stderr.write(usage)
            sys.exit()

    m = mask_unrelated(u=u, d=d, lc=lc, o=o)

    if a is not None and b is not None:
        m(a.split(), b.split())

    if fa is not None and fb is not None:
        with open(fa) as f1, open(fb) as f2:
            for a, b in zip(f1, f2):
                m(a.split(), b.split())

    n_sents, n_tok_a, n_tok_b, n_unr_a, n_unr_b = m.stats()
    perc_a = 0.0 if n_tok_a==0 else 100.0*n_unr_a/n_tok_a
    perc_b = 0.0 if n_tok_b==0 else 100.0*n_unr_b/n_tok_b
    sys.stderr.write('{} sents found, n_tokens (a={},b={}), n_unrelated (a={:.1f}%,b={:.1f}%)\n'.format(n_sents, n_tok_a, n_tok_b, perc_a, perc_b))


