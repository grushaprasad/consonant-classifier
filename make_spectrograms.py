import librosa, tgt
import numpy as np, math
import glob, pickle, re, sys
import os

vowels = {
    'a': 'AA',
    'i': 'IY',
    'u': 'UW',
    'ae': 'AE'
}

stops = ['D', 'G']
pattern = '^['+ ''.join(stops) +'][AEIOU]'
lab_to_int = {
    'D': 0,
    'G': 1
}


#stop_cats = ['P', 'B', 'T', 'D', 'K', 'G']

# Kaldi description:
# "number of frames in the file: typically 25 ms frames shifted by 10ms each time"

# spectrogram parameters
sr_target   = 11025 # 11.025 kHz of Stephens & Holt (2011)
n_fft = int((sr_target*20)/1000)  # makes the window size roughly 25 ms. 
hop_length = int((sr_target*10)/1000) # shifts the window by roughly 10 ms
n_mels      = 64    # librosa default is 128
eps         = 1.0e-8
max_dur = 1.16 #seconds as computed by get_max_dur

padding = 'unpadded'

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
    if padding == 'padded':
        snd = add_silence(snd, sr_target, dur, max_dur)
    melspec = librosa.feature.melspectrogram(y=snd, sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
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
            vowel = vowel_interval.text[0:2]
            if vowel in vowels.values():
                logmelspec = make_spectrogram(wavfile, start_time, dur)
                results.append((wavfile,logmelspec, syll, stop, vowel, lab_to_int[stop]))  # [spectrogram, word, consonant, vowel]
            #results.append((logmelspec, stop))
            #print (results)
            #sys.exit(0)
    print(len(results))
    return results

# Make spectrograms for all natural recordings, 
    ## grouped by speaker
if 0:
    maindir = './ChodroffWilson2014'
    gridfiles = glob.glob(maindir+'/textgrid/*.TextGrid')

    for i,gridfile in enumerate(gridfiles):

        print('Processing %s of %s'%(i+1, len(gridfiles)))
        wavfile = re.sub('/textgrid/', '/wav/', gridfile)
        wavfile = re.sub('.TextGrid', '_edited.wav', wavfile)
        results = make_spectrograms(wavfile, gridfile)

        pklfile = re.sub('/textgrid/', '/spectrograms/%s/%smels/'%(padding,n_mels), gridfile)
        pklfile = re.sub('.TextGrid', '.pkl', pklfile)

        #print(pklfile)

        with open(pklfile, 'wb') as f:
            pickle.dump(results, f)
        #sys.exit(0)


if 0:
    maindir = './SyntheticDG/wav/'
    #results = []
    ends = []
    middle = []

    wavfiles = glob.glob(maindir+'*.wav')

    for f in wavfiles:

        everything = []

        if len(re.findall('\d+', f)) > 0:  #ignores other files
            filenum = int(re.findall('\d+', f)[0])
            f_name = os.path.basename(f)
            if f_name[1] == '_':
                vowel = vowels[f_name[0:1]]
            else:
                vowel = vowels[f_name[0:2]]

            if filenum < 11:
                lab = 'D'
            else:
                lab = 'G'
            
            logmelspec = make_spectrogram(f)
            everything.append((f, logmelspec, 'NA', lab, vowel,lab_to_int[lab]))

            if filenum < 6 or filenum > 15:
                ends.append((f, logmelspec, 'NA', lab, vowel,lab_to_int[lab]))
            else:
                middle.append((f, logmelspec, 'NA', lab, vowel,lab_to_int[lab]))


            pklfile = re.sub('/wav/', '/spectrograms/%s/%smels/%s/'%(padding,n_mels,vowel), f)
            pklfile = re.sub('.wav', '.pkl', pklfile)

            #print(pklfile)


            with open(pklfile, 'wb') as f:
                pickle.dump(everything, f)



## Make spectrograms for precursors
if 1:
    standard_freq = 2300.0
    tone_dur = 70.0/1000.0 # 70 ms duration for each tone
    isi_dur = 30.0/1000.0 # 30 ms silence between successive tones
    sil_dur = 50.0/1000.0 # 50 ms silence between standard tone and speech
    ramp_dur = 5.0/1000.0
    sr = 11025

    offset = 0.098 
    duration = 0.365

    rms_target = 0.06362659

    num_conds = 4
    num_trials_per_cond = 10

    stim_files = ['a_dg09_CV.wav', 'a_dg10_CV.wav', 'a_dg11_CV.wav', 'a_dg12_CV.wav', 'a_dg13_CV.wav','a_dg14_CV.wav','a_dg15_CV.wav','a_dg16_CV.wav']

    def rms_amplitude(x):
        rms = np.sqrt(np.mean(x**2.0))
        return rms

    def scale_rms(x, rms_target):
        rms_x = rms_amplitude(x)
        s = (rms_target / rms_x)
        y = s*x
        return y

    def create_precursor_freqs(l1,l2):
        np.random.shuffle(l1)
        np.random.shuffle(l2)
        return(np.concatenate((l1, l2),0))

    def create_precursors(l, tone_sr, tone_dur):   
        linear_ramp = np.linspace(0.0, 1.0, int(ramp_dur*tone_sr))
        mask = np.ones(int((tone_dur - 2.0*ramp_dur)*tone_sr))
        mask = np.concatenate((linear_ramp, mask, linear_ramp[::-1]))
        standard_tone = mask*librosa.core.tone(standard_freq, tone_sr, duration=tone_dur)[0:771] 


        sil = np.zeros(int(sil_dur * tone_sr))
        isi = np.zeros(int(isi_dur * tone_sr))

        precursor_list = [0]*len(l)
        for i, item in enumerate(l):

            curr_tones = [mask*librosa.core.tone(freq, tone_sr, duration=tone_dur)[0:771] for freq in item]
            curr_precursor = np.concatenate([np.concatenate((tone, isi)) for tone in curr_tones] + [standard_tone, sil])
            precursor_list[i] = scale_rms(curr_precursor, rms_target)
        return(precursor_list)

    low = np.arange(800.0, 1800.0+100.0, 100.0) # mean 1300
    mid = np.arange(1800.0, 2800.0+100.0, 100.0) # mean 2300
    high = np.arange(2800.0, 3800.0+100.0, 100.0) # mean 3300

    lowmid = [create_precursor_freqs(low, mid) for i in range(num_trials_per_cond*len(stim_files))]
    midlow = [create_precursor_freqs(mid, low) for i in range(num_trials_per_cond*len(stim_files))]
    highmid = [create_precursor_freqs(high, mid) for i in range(num_trials_per_cond*len(stim_files))]
    midhigh = [create_precursor_freqs(mid, high) for i in range(num_trials_per_cond*len(stim_files))]

    #lowmid = [create_precursor_freqs(low, mid)]

    # This is taking just just the first precursor. In the future maybe save the log melspec for a bunch of precursors?
    lowmid_precursors = create_precursors(lowmid, sr, tone_dur)
    lowmid_melspec = librosa.feature.melspectrogram(y=lowmid_precursors[0], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    lowmid_logmelspec = np.log(lowmid_melspec + eps)

    midlow_precursors = create_precursors(midlow, sr, tone_dur)
    midlow_melspec = librosa.feature.melspectrogram(y=midlow_precursors[0], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    midlow_logmelspec = np.log(midlow_melspec + eps)

    highmid_precursors = create_precursors(highmid, sr, tone_dur)
    highmid_melspec = librosa.feature.melspectrogram(y=highmid_precursors[0], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    highmid_logmelspec = np.log(highmid_melspec + eps)

    midhigh_precursors = create_precursors(midhigh, sr, tone_dur)
    midhigh_melspec = librosa.feature.melspectrogram(y=midhigh_precursors[0], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    midhigh_logmelspec = np.log(midhigh_melspec + eps)


    pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'lowmid.pkl'
    with open(pklfile, 'wb') as f:
        pickle.dump(lowmid_logmelspec, f)

    pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'midlow.pkl'
    with open(pklfile, 'wb') as f:
        pickle.dump(midlow_logmelspec, f)

    pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'highmid.pkl'
    with open(pklfile, 'wb') as f:
        pickle.dump(highmid_logmelspec, f)

    pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'midhigh.pkl'
    with open(pklfile, 'wb') as f:
        pickle.dump(midhigh_logmelspec, f)



   