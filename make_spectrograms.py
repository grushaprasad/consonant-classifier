import librosa, tgt
import numpy as np, math
import glob, pickle, re, sys
import os


stops = ['D', 'G']
pattern = '^['+ ''.join(stops) +'][AEIOU]'
#stop_cats = ['P', 'B', 'T', 'D', 'K', 'G']

# spectrogram parameters
sr_target   = 11025 # 11.025 kHz of Stephens & Holt (2011)
n_fft       = 512
hop_length  = int(n_fft/2)
n_mels      = 64    # librosa default is 128
eps         = 1.0e-8
max_dur = 1.16 #seconds as computed by get_max_dur


# gets the maximum duration of all 

def add_silence(sound, sr, sound_dur, max_dur):
    num_zeros = int(math.ceil((max_dur-sound_dur) * sr))
    return(np.append(sound, num_zeros*[0]))

def get_mu_sd(specs):
    #print(type(specs))
    spec_all = np.concatenate(specs,0)
    # spec_all[spec_all == 0] = np.nan  #replace 0 with nan
    # spec_all = np.ma.array(spec_all, mask=np.isnan(spec_all)) #mask nan

    mu = np.mean(spec_all,0)  # takes mean ignoring nan across filters
    sd = np.sqrt(np.var(spec_all, 0))
    return(mu,sd)

# make spectrogam from soundfile
def make_spectrogram(wavfile, offset=0.0, duration=None, log=True):
    snd,sr  = librosa.core.load(wavfile, sr=sr_target, offset=offset, duration=duration) 
    dur = librosa.get_duration(y=snd, sr=sr_target)
    snd_padded = add_silence(snd, sr_target, dur, max_dur)
    melspec = librosa.feature.melspectrogram(y=snd_padded, sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    logmelspec = np.log(melspec + eps)
    if log:
        return(logmelspec)
    else:
        return(melspec)


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
            results.append((logmelspec, syll, stop, vowel))  # [spectrogram, word, consonant, vowel]
            #results.append((logmelspec, stop))
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

if 0:
    maindir = './SyntheticDG/wav/'
    #results = []
    ends = []
    middle = []
    everything = []

    wavfiles = glob.glob(maindir+'*.wav')

    for f in wavfiles:
        if len(re.findall('\d+', f)) > 0:  #ignores other files
            filenum = int(re.findall('\d+', f)[0])
            if f[1] == '_':
                vowel = f[0:1]
            else:
                vowel = f[0:2]

            if filenum < 11:
                lab = 'D'
            else:
                lab = 'G'
            
            logmelspec = make_spectrogram(f)
            everything.append((logmelspec, 'NA', lab, vowel, f))

            if filenum < 6 or filenum > 15:
                ends.append((logmelspec, 'NA', lab, vowel, f))
            else:
                middle.append((logmelspec, 'NA', lab, vowel, f))

            #results.append((logmelspec, lab))

    pklfile = './SyntheticDG/spectrograms/' + 'SyntheticDG_%s_ends.pkl'%n_mels
    with open(pklfile, 'wb') as f:
            pickle.dump(ends, f)


    pklfile = './SyntheticDG/spectrograms/' + 'SyntheticDG_%s_middle.pkl'%n_mels
    with open(pklfile, 'wb') as f:
            pickle.dump(middle, f)

    pklfile = './SyntheticDG/spectrograms/' + 'SyntheticDG_%s_all.pkl'%n_mels
    with open(pklfile, 'wb') as f:
            pickle.dump(everything, f)



## Make spectrograms subtracted from precursor

ntone = 21 # 21 sine-wav tones with unique frequencies that depend on condition:
low_mean = np.arange(1300.0, 2300.0+50.0, 50.0) # mean = 1800 Hz
high_mean = np.arange(2300.0, 3300.0+50.0, 50.0) # mean = 2800 Hz
mid_mean_low_var = np.arange(1800.0, 2800.0+50.0, 50.0) # mean = 2300 Hz
mid_mean_high_var = np.arange(1300.0, 3300.0+50.0, 100.0) # mean = 2300 Hz
standard_freq = 2300.0 # standard tone at 2300 Hz
tone_dur = 70.0/1000.0 # 70 ms duration for each tone
isi_dur = 30.0/1000.0 # 30 ms silence between successive tones
sil_dur = 50.0/1000.0 # 50 ms silence between standard tone and speech
ramp_dur = 5.0/1000.0 # 5 ms onset and offset ramp for each tone
sr = 10000.0 # 10 kHz sampling rate

linear_ramp = np.linspace(0.0, 1.0, int(ramp_dur*sr))
mask = np.ones(int((tone_dur - 2.0*ramp_dur)*sr))
mask = np.concatenate((linear_ramp, mask, linear_ramp[::-1]))

low_tones = [mask*librosa.core.tone(freq, sr, duration=tone_dur)[0:700] for freq in low_mean]
high_tones = [mask*librosa.core.tone(freq, sr, duration=tone_dur)[0:700] for freq in high_mean]
standard_tone = mask*librosa.core.tone(standard_freq, sr, duration=tone_dur)[0:700]


np.random.shuffle(low_tones)
np.random.shuffle(high_tones)
isi = np.zeros(int(isi_dur * sr))
sil = np.zeros(int(sil_dur * sr))

low_condition = np.concatenate(
    [np.concatenate((tone, isi)) for tone in low_tones] + [standard_tone,sil])
high_condition = np.concatenate(
    [np.concatenate((tone, isi)) for tone in high_tones] + [standard_tone, sil])

melspec_low = librosa.feature.melspectrogram(y=low_condition, sr=sr, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
logmelspec_low = np.log(melspec_low + eps)
mu_low, _ = get_mu_sd(logmelspec_low)

melspec_high = librosa.feature.melspectrogram(y=high_condition, sr=sr, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
logmelspec_high = np.log(melspec_high + eps)
mu_high, _ = get_mu_sd(logmelspec_high)


if 1:
    maindir = './SyntheticDG/wav/'
    #results = []
    low_subtracted = []
    high_subtracted = []
    #maxes = []

    wavfiles = glob.glob(maindir+'*.wav')

    for f in wavfiles:
        if len(re.findall('\d+', f)) > 0:  #ignores other files
            filenum = int(re.findall('\d+', f)[0])
            if f[1] == '_':
                vowel = f[0:1]
            else:
                vowel = f[0:2]

            if filenum < 11:
                lab = 'D'
            else:
                lab = 'G'
            
            logmelspec = make_spectrogram(f)
            #everything.append((logmelspec, 'NA', lab, vowel, f))

            # melspec = make_spectrogram(f, log= False)
            # #maxes.append(np.amin(melspec))

            logmelspec_high_subtracted = logmelspec - mu_high
            #melspec_high_subtracted[melspec_high_subtracted < 0] = 0
            #logmelspec_high_subtracted = np.log(melspec_high_subtracted + eps)
            high_subtracted.append((logmelspec_high_subtracted, 'NA', lab, vowel, f))

            logmelspec_low_subtracted = logmelspec - mu_low
            #melspec_low_subtracted[melspec_low_subtracted < 0] = 0
            #logmelspec_low_subtracted = np.log(melspec_low_subtracted + eps)
            low_subtracted.append((logmelspec_low_subtracted, 'NA', lab, vowel, f))

            logmelspec = make_spectrogram(f)




    pklfile = './SyntheticDG/spectrograms/' + 'SyntheticDG_%s_all_high_subtracted.pkl'%n_mels
    with open(pklfile, 'wb') as f:
            pickle.dump(high_subtracted, f)


    pklfile = './SyntheticDG/spectrograms/' + 'SyntheticDG_%s_all_low_subtracted.pkl'%n_mels
    with open(pklfile, 'wb') as f:
            pickle.dump(low_subtracted, f)



