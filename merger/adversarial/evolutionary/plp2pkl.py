# -*- coding: utf-8 -*-
import pickle
from sys import argv
from struct import unpack, pack
from pkl2feats import process_pickle, merge_txts
import numpy
import os
import sys
import audacity
import progressbar
import argparse
import numpy as np

from meta import paths, id_from_cued

#def read_leap():
 #   metadata = {}
  #  with open(paths['Coding']) as fin:
   #     for line in fin.readlines():
    #        wav_path = line.split()[0].split('/')[-1].split('_')
     #       metadata[line.split()[1].split('/')[-1].split('-en')[0]] = {'L1': lang_codes[wav_path[1]], 'Gender': wav_path[3]}
    #return metadata

#meta_leap_dict = read_leap()

def read_metadata():
    metadata = {}
    for metadata_dir in ['Metadata','LSMetadata','LSMetadata2','LeapMetadata','SellMetadata']:
        with open(paths[metadata_dir]) as fin:
            for line in fin.readlines()[1:]:
                l = line.split('\t')[1:]
                if all([j != '.' for j in l]) or 'Sell' in metadata_dir:
                    metadatum = {}
                    columns = ['Country', 'L1', 'Gender', 'YoB','Version','CEFR']
                    for k in range(len(columns)):
                        metadatum[columns[k]] = l[k+1]
                    nm = np.mean([float(i) for i in l[7:-1] if i != 'M' and i != 'U' and i != 'T' and i != '.'])
                    metadatum['Scores'] = [float(i) if (i != 'M' and i != 'T' and i != 'U' and i != '.'
                                                        ) else nm for i in l[7:-1]]
                    metadata[l[0]] = metadatum
    return metadata

metadata_dict = read_metadata()

def get_meta(utt):
    u = '-'.join([utt.split('-')[0],'XXXXX']+utt.split('-')[2:4])
    #if u in meta_leap_dict:
     #   return meta_leap_dict[u]['L1'], meta_leap_dict[u]['Gender'], -1
    cued = '-'.join(utt.split('-')[:2])
    if cued[-1] == 'X':
        cued = cued[:-1]+'1'
    spk = id_from_cued[cued]
    if spk not in metadata_dict:
        spk = spk+'_can'
    #l1 = lang_codes[spk.split('_')[1]] if (len(spk.split('_')) > 1 and spk.split('_')[1] in lang_codes) else metadata_dict[spk]['L1']
    l1 = metadata_dict[spk]['L1']
    gender = metadata_dict[spk]['Gender'] #if spk in metadata_dict else spk.split('_')[3].upper()
    secs = ['SA0','SB0','SC0','SD0','SE0']
    sec = None
    for j in range(5):
        if secs[j] in utt:
            sec = j
    scores = metadata_dict[spk]['Scores']
    score = sum(scores)*1.0/len(scores) if sec is None else scores[sec]
    return l1, gender, score

def get_phones(alphabet):
    if alphabet == 'arpabet':
        vowels = ['aa', 'ae', 'eh', 'ah', 'ea', 'ao', 'ia', 'ey', 'aw', 'ay', 'ax', 'er', 'ih', 'iy',
          'uh', 'oh', 'oy', 'ow', 'ua', 'uw']
        consonants = ['el', 'ch', 'en', 'ng', 'sh', 'th', 'zh', 'w', 'dh', 'hh', 'jh', 'em', 'b', 'd',
              'g', 'f', 'h', 'k', 'm', 'l', 'n', 'p', 's', 'r', 't', 'v', 'y', 'z']+['sil']
        phones = vowels+consonants
        return vowels, consonants, phones
    if alphabet == 'graphemic':
        vowels = ['a','e','i','o','u']
        consonants = ['b','c','d','f','g','h','j','k','l','m','n','o','p','q','r','s','t','v','w','x','y','z']+['sil']
        phones = vowels+consonants
        return vowels, consonants, phones    
    with open(alphabet,'r') as fin:
        return None,None, [l.split()[0] for l in fin.readlines()]

LPC = 1
LPCREFC = 2
LPCEPSTRA = 3
LPCDELCEP = 4
IREFC = 5
MFCC = 6
FBANK = 7
MELSPEC = 8
USER = 9
DISCRETE = 10
PLP = 11

_E = 0000100 # has energy
_N = 0000200 # absolute energy supressed
_D = 0000400 # has delta coefficients
_A = 0001000 # has acceleration (delta-delta) coefficients
_C = 0002000 # is compressed
_Z = 0004000 # has zero mean static coefficients
_K = 0010000 # has CRC checksum
_O = 0020000 # has 0th cepstral coefficient
_V = 0040000 # has VQ data
_T = 0100000 # has third differential coefficients

def hopen(f, mode=None, veclen=13):
    """Open an HTK format feature file for reading or writing.
    The mode parameter is 'rb' (reading)."""
    if mode is None:
        if hasattr(f, 'mode'):
            mode = f.mode
        else:
            mode = 'rb'
    if mode in ('r', 'rb'):
        return HTKFeat_read(f) # veclen is ignored since it's in the file
    else:
        raise Exception, "mode must be 'r'or  'rb'"

class HTKFeat_read(object):
    "Read HTK format feature files"
    def __init__(self, filename=None):
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        if (filename != None):
            self.open(filename)

    def __iter__(self):
        self.fh.seek(12,0)
        return self

    def open(self, filename):
        self.filename = filename
        self.fh = file(filename, "rb")
        self.readheader()

    def readheader(self):
        self.fh.seek(0,0)
        spam = self.fh.read(12)
        self.nSamples, self.sampPeriod, self.sampSize, self.parmKind = \
                       unpack(">IIHH", spam)
        # Get coefficients for compressed data
        if self.parmKind & _C:
            self.dtype = 'h'
            self.veclen = self.sampSize / 2
            if self.parmKind & 0x3f == IREFC:
                self.A = 32767
                self.B = 0
            else:
                self.A = numpy.fromfile(self.fh, 'f', self.veclen)
                self.B = numpy.fromfile(self.fh, 'f', self.veclen)
                if self.swap:
                    self.A = self.A.byteswap()
                    self.B = self.B.byteswap()
        else:
            self.dtype = 'f'    
            self.veclen = self.sampSize / 4
        self.hdrlen = self.fh.tell()

    def seek(self, idx):
        self.fh.seek(self.hdrlen + idx * self.sampSize, 0)
        
    def getinterval(self, start, end):
        self.seek(start)
        return self.next_n(end-start)
         
    def next_n(self, n):
        data = numpy.fromfile(self.fh, self.dtype, self.veclen*n)
        data = data.reshape(len(data)/self.veclen, self.veclen)
        if self.swap:
            data = data.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            data = (data.astype('f') + self.B) / self.A
        return data

    def next(self):
        vec = numpy.fromfile(self.fh, self.dtype, self.veclen)
        if len(vec) == 0:
            raise StopIteration
        if self.swap:
            vec = vec.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            vec = (vec.astype('f') + self.B) / self.A
        return vec

    def readvec(self):
        return self.next()

    def getall(self,cs):
        self.seek(0)
        data = numpy.fromfile(self.fh, self.dtype)
        if self.parmKind & _K: # Remove and ignore checksum
            if cs:
                data = data[:-1]
        data = data.reshape(len(data)/self.veclen, self.veclen)
        if self.swap:
            data = data.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            data = (data.astype('f') + self.B) / self.A
        return data

def read_scp(scp_file):
    scp_dict = {}
    fin = open(scp_file, 'r')
    for line in fin.readlines():
        if '=' in line:
            r = line.split('=')[1]
            scp_dict[line.split('=')[0]] = (r.split('[')[0], int(r.split('[')[1].split(',')[0]), int(r.split(',')[1].split(']')[0]))
    fin.close()
    return scp_dict

def read_mlf(mlf_file):
    fin = open(mlf_file, 'r')
    lines = [' '.join(l.split()[:3])+'\n' if len(l.split()) > 3 else l for l in fin.readlines()[1:]]
    lines = [l[:-2]+'\n' if (len(l) > 2 and l[-2] == '.') else l for l in lines]
    subfiles = {s.split('\n')[0].replace('"','').replace('*/','').replace('.rec','.lab'): s for s in ''.join(lines).split('.\n')[:-1]}
    fin.close()
    return subfiles

def read_mlf_sub(sub, wsub, scp_dct, ex, wlab_file, mlab_file, feats, utt_inds, alphabet,no_meta=False):
    vowels, consonants, phones = get_phones(alphabet)
    mlf_file = sub.split('\n')
    wmlf_file = wsub.split('\n')
    mlf_header = mlf_file[0]
    utt = mlf_header.replace('"','').split('.lab' if '.lab' in mlf_header else '.rec')[0].split('/')[-1]
    utt_name = '-'.join(utt.split('-')[:4])
    spk = '-'.join(utt.split('-')[:2]).replace('"','')
    spk_index = ex['speaker'].index(spk)
    ts1 = int(utt.split('-')[-1].split('_')[-2])
    ts2 = int(utt.split('-')[-1].split('_')[-1])
    #print(utt,len(ex['utterance'][spk_index]))
    ex['utterance'][spk_index][utt_inds[utt]] = utt_name
    if not no_meta:
        l1, gender, score = get_meta(utt_name)
        ex['l1'][spk_index] = l1
        ex['gender'][spk_index] = gender
        ex['score'][spk_index][utt_inds[utt]] = score
    ex['start_time'][spk_index][utt_inds[utt]] = ts1
    ex['end_time'][spk_index][utt_inds[utt]] = ts2
    
    file_name, start_time, end_time = scp_dct[utt+'.plp']
    fin = hopen(file_name)
    lines = [mlf_file[f].split()[:3] for f in range(1,len(mlf_file)-1) if 'na' not in mlf_file[f].split()[2] and (f == len(mlf_file)-2 or mlf_file[f].split()[2] not in ['sil','sp'] or mlf_file[f+1].split()[2] not in ['sil','sp'])]
    wlines = [f.split()[:3] for f in wmlf_file[1:-1] if 'na' not in f.split()[2] and f.split()[2] not in ['sil','sp','<s>']]
    boundaries = [j for j in range(len(lines)) if lines[j][2] in ['sil','sp']]
    l = [(int(line[0])/100000, int(line[1])/100000, line[2].split('-')[-1].split('^' if '^' in line[2] else '+')[0]) for line in lines]
    #print(wlines,boundaries,utt)
    #print(len(wlines),len(boundaries))
    if len(wlines) != len(boundaries)-1:
        print(len(wlines),len(boundaries)-1)
        print([f[2] for f in wlines])
        raise ValueError('Number of words does not match number of silences for '+mlf_header)
    word_indices = [i for i in range(len(boundaries)-1) if wlines[i][2] != '%HESITATION%' and boundaries[i+1]-boundaries[i] -2 <= 15]
    ex['word'][spk_index][utt_inds[utt]] = [wlines[i][2] for i in word_indices]
    for feat in feats:
        ex[feat][spk_index][utt_inds[utt]] = [None]*len(word_indices)
    for j in range(len(word_indices)):
        i = word_indices[j]
        word = ex['word'][spk_index][utt_inds[utt]][j]
        for feat in feats:
            ex[feat][spk_index][utt_inds[utt]][j] = [None]*(boundaries[i+1]-boundaries[i]-1)
        for k in range(boundaries[i]+1, boundaries[i+1]):
            index = k-boundaries[i]-1
            start_time, end_time, ph = l[k]
            ex['phone'][spk_index][utt_inds[utt]][j][index] = phones.index(ph.lower().replace('ehr','eh').replace('ihr','ih').replace('iyr','iy').replace('wh','w'))
            ex['ph_start_time'][spk_index][utt_inds[utt]][j][index] = start_time
            ex['ph_end_time'][spk_index][utt_inds[utt]][j][index] = end_time
            plp = fin.getinterval(start_time, end_time)
            dplp = [plp[feat+1]-plp[feat-1] for feat in range(1,len(plp)-1)]
            d0 = plp[1]-(fin.getinterval(start_time-1, start_time)[0] if start_time > 0 else plp[0])
            dprev = (plp[0]-fin.getinterval(start_time-2, start_time-1)[0]) if start_time > 1 else None
            dend = plp[-1]-plp[-2]
            dplp = [d0] + dplp + [dend]
            ddplp = [dplp[feat+1]-dplp[feat-1] for feat in range(1,len(dplp)-1)]
            dd0 = ddplp[0] if dprev is None else (d0 - dprev)
            ddend = ddplp[-1]
            ddplp = [dd0] + ddplp + [ddend]
            dplp = np.array(dplp)[:,1:]
            ddplp = np.array(ddplp)[:,1:]
            plp = np.array(plp)[:,1:]
            plp = np.concatenate((plp, dplp, ddplp),1)
            ex['plp'][spk_index][utt_inds[utt]][j][index] = plp
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-WLAB', required=True, help='Word MLF file') 
    parser.add_argument('-MLAB', required=True, help='Aligned monophone MLF file')
    parser.add_argument('-SCP', required=True, help='SCP file') 
    parser.add_argument('-ALPHABET', required=True, help='name of stored alphabet: arpabet, sampa, graphemes or path to custom alphabet') 
    parser.add_argument('-TSET', required=True, help='Name of dataset (for producing output file) e.g. BLXXXeval3') 
    parser.add_argument('-NO_SCORES', action='store_true', help='Don\'t load scores') 
    parser.add_argument('-NO_META', action='store_true', help='Don\'t load any metadata')
    parser.add_argument('-MSGPACK', action='store_true', help='Use msgpack instead of pickle')
    args = parser.parse_args()
 
    wlab_mlf = args.WLAB
    mlab_mlf = args.MLAB
    scp_file = args.SCP
    tset = args.TSET

    if os.path.isdir('CMDs'):
        with open('CMDs/plp2pkl.cmds', 'a') as f:
            f.write(' '.join(sys.argv)+'\n')
    else:
        os.mkdir('CMDs')
        with open('CMDs/plp2pkl.cmds', 'a') as f:
            f.write(' '.join(sys.argv)+'\n')

    scp_dict = read_scp(scp_file)
    context_feats = ['utterance','spk_index','start_time','end_time'] + ([] if args.NO_META else ['l1','gender','score'])
    sequence_feats = ['phone','ph_start_time','ph_end_time','plp']
    mlf = read_mlf(mlab_mlf)
    wmlf = read_mlf(wlab_mlf)
    bar = progressbar.ProgressBar(maxval=len(mlf))
    bar.start()
    utt_inds = {}
    spks = {}
    for utt in mlf:
        spk = '-'.join(utt.split('-')[:2]).replace('"','').replace('*/','')
        spks[spk] = 1 if spk not in spks else (spks[spk]+1)
        utt_inds[utt.split('.')[0].replace('"','').replace('*/','')] = spks[spk]-1
    ex = {'speaker': [k for k in spks]}
    for feat in context_feats+sequence_feats+['word']:
        ex[feat] = [[None]*spks[s] for s in spks]
    j=0
    k=0
    keys = [m for m in mlf if m in wmlf]

    for key in keys:
        j+=1
        try:
            read_mlf_sub(mlf[key], wmlf[key], scp_dict, ex, wlab_mlf, mlab_mlf, sequence_feats, utt_inds, args.ALPHABET,
                         args.NO_META)
        except pickle.PickleError as e:
            print(e)
            k += 1
        bar.update(j)
    bar.finish()
    indices = range(0,len(ex['speaker']),2000)+[-1]
    for j in range(len(indices)-1):
        obj = {feat: ex[feat][indices[j]:indices[j+1]] for feat in ex}
        process_pickle(obj, tset+('-'+str(j) if len(indices) > 2 else '')+'.pkl',
                       tset+('_'+str(j) if len(indices) > 2 else '')+'.txt''.',
                       args.ALPHABET, args.NO_SCORES, args.MSGPACK)
    print(k)
    merge_txts('.', args.TSET)
