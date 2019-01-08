import librosa, tgt
import numpy as np
import glob, pickle, re, sys
import os


stops = ['D', 'G']
pattern = '^['+ ''.join(stops) +'][AEIOU]'
#stop_cats = ['P', 'B', 'T', 'D', 'K', 'G']

# spectrogram parameters
sr_target   = 11025 # 11.025 kHz of Stephens & Holt (2011)
n_fft       = 512
hop_length  = int(n_fft/2)
n_mels      = 128    # librosa default is 128
eps         = 1.0e-8


# make spectrogam from soundfile
def make_spectrogram(wavfile, offset=0.0, duration=None):
    snd,sr  = librosa.core.load(wavfile, sr=sr_target, offset=offset, duration=duration)
    melspec = librosa.feature.melspectrogram(y=snd, sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    logmelspec = np.log(melspec + eps)
    return logmelspec


 
# make spectrograms for all stop intervals in textgrid
def make_spectrograms(wavfile, gridfile):
    if gridfile is None:
        return spectrogram(wavfile)
    grid    = tgt.io.read_textgrid(gridfile)
    words   = grid.get_tier_by_name('word')
    #print(words)
    phons   = grid.get_tier_by_name('phone')

    #pattern = '^['+ ''.join(stops) +'][AEIOU][AEIOU]T$'
    #print(pattern)
    word_indx = 0
    results = []
    for word in words:
        syll = word.text
        if re.match(pattern, syll):
            word_indx += 1
            start_time, end_time = word.start_time, word.end_time
            dur = (end_time - start_time)
            
            phon = phons.get_annotations_between_timepoints(start_time, end_time)
            indx = [i for i,x in enumerate(phon) if x.text in stops][0]
            stop_interval, vowel_interval = phon[indx], phon[indx+1]
            start_time = stop_interval.start_time - 0.025
            end_time  = vowel_interval.end_time

            syll = syll+str(word_indx)
            stop = stop_interval.text
            vowel = vowel_interval.text
            logmelspec = make_spectrogram(wavfile, start_time, dur)
            #results.append((logmelspec, syll, stop, vowel))  # [spectrogram, word, consonant, vowel]
            results.append((logmelspec, stop))
            #print (results)
            #sys.exit(0)
 
    return results

# make spectrograms for all natural recordings, 
# grouped by speaker
if 0:
    maindir = './ChodroffWilson2014'
    gridfiles = glob.glob(maindir+'/textgrid/*.TextGrid')

    for i,gridfile in enumerate(gridfiles):
        print('Processing %s of %s'%(i+1, len(gridfiles)))
        wavfile = re.sub('/textgrid/', '/wav/', gridfile)
        wavfile = re.sub('.TextGrid', '_edited.wav', wavfile)
        results = make_spectrograms(wavfile, gridfile)

        pklfile = re.sub('/textgrid/', '/spectrograms/%s/'%n_mels, gridfile)
        pklfile = re.sub('.TextGrid', '.pkl', pklfile)

        with open(pklfile, 'wb') as f:
            pickle.dump(results, f)
        #sys.exit(0)


#make spectrograms for all synthetic stimuli
# if 0:
#     maindir = '/Users/colin/Dropbox/GrushaSecondProject/StephensHolt2011'
#     wavfiles = glob.glob(maindir+'/stimuli/CV_stimuli/*/*/*.wav')
#     results = []
#     for wavfile in wavfiles:
#         logmelspec = make_spectrogram(wavfile)
#         stim = re.sub('.*/', '', wavfile)
#         stim = re.sub('.wav', '', stim)
#         results.append((logmelspec, stim))

#     pklfile = maindir+'/dat/CV_stimuli.pkl'
#     with open(pklfile, 'wb') as f:
#         pickle.dump(results, f)


if 1:
    maindir = './SyntheticDG/wav/'
    #results = []
    ends = []
    middle = []

    wavfiles = glob.glob(maindir+'*.wav')

    for f in wavfiles:
        if len(re.findall('\d+', f)) > 0:  #ignores other files
            filenum = int(re.findall('\d+', f)[0])
            if filenum < 11:
                lab = 'D'
            else:
                lab = 'G'
            

            logmelspec = make_spectrogram(f)

            if filenum < 6 or filenum > 15:
                ends.append((logmelspec, lab))
            else:
                middle.append((logmelspec, lab))

            #results.append((logmelspec, lab))

    pklfile = './SyntheticDG/spectrograms/' + 'SyntheticDG_%s_ends.pkl'%n_mels
    with open(pklfile, 'wb') as f:
            pickle.dump(ends, f)


    pklfile = './SyntheticDG/spectrograms/' + 'SyntheticDG_%s_middle.pkl'%n_mels
    with open(pklfile, 'wb') as f:
            pickle.dump(middle, f)

    # for file in os.listdir(os.fsencode(maindir)):
    #     filename = maindir + os.fsdecode(file)
    #     if len(re.findall('\d+', filename)) > 0:  #ignores other files
    #         filenum = int(re.findall('\d+', filename)[0])
    #         if filenum < 11:
    #             lab = 'D'
    #         else:
    #             lab = 'G'
            

    #         logmelspec = make_spectrogram(filename)

    #         if filenum < 6 or filenum > 15:
    #             ends.append((logmelspec, lab))
    #         else:
    #             middle.append((logmelspec, lab))

    #         #results.append((logmelspec, lab))

    # pklfile = './SyntheticDG/spectrograms/' + 'SyntheticDG_%s_ends.pkl'%n_mels
    # with open(pklfile, 'wb') as f:
    #         pickle.dump(ends, f)


    # pklfile = './SyntheticDG/spectrograms/' + 'SyntheticDG_%s_middle.pkl'%n_mels
    # with open(pklfile, 'wb') as f:
    #         pickle.dump(middle, f)




