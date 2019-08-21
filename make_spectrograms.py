import librosa, tgt
import numpy as np, math
import glob, pickle, re, sys
import os
import itertools

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
n_mels      = 20    # librosa default is 128
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

    def create_precursor_freqs(l1,l2=[]):
        np.random.shuffle(l1)
        np.random.shuffle(l2)
        return(np.concatenate((l1, l2),0))

    def create_precursors(l, tone_sr, tone_dur, num_standard=1, num_silence=1, num_standard_isi=0):   
        linear_ramp = np.linspace(0.0, 1.0, int(ramp_dur*tone_sr))
        mask = np.ones(int((tone_dur - 2.0*ramp_dur)*tone_sr))
        mask = np.concatenate((linear_ramp, mask, linear_ramp[::-1]))
        standard_tone = mask*librosa.core.tone(standard_freq, tone_sr, duration=tone_dur)[0:771] 


        sil = np.zeros(int(sil_dur * tone_sr))
        isi = np.zeros(int(isi_dur * tone_sr))

        #print(([standard_tone] + [isi])*num_standard)

        precursor_list = [0]*len(l)
        for i, item in enumerate(l):

            curr_tones = [mask*librosa.core.tone(freq, tone_sr, duration=tone_dur)[0:771] for freq in item]


            #curr_precursor = np.concatenate([np.concatenate((tone, isi)) for tone in curr_tones] + [standard_tone]*num_standard + [isi]*num_standard_isi + [sil]*num_silence)

            curr_precursor = np.concatenate([np.concatenate((tone, isi)) for tone in curr_tones] + ([standard_tone] + [isi])*num_standard + [sil]*num_silence)

            x = np.concatenate([np.concatenate((tone, isi)) for tone in curr_tones] + [standard_tone]*num_standard + [isi]*num_standard + [sil]*num_silence)

            #print(list(itertools.chain(*zip(standard_tone, isi))))

            #curr_precursor = np.concatenate([np.concatenate((tone, isi)) for tone in curr_tones] + list(itertools.chain(*zip(standard_tone, isi)))*num_standard + [sil]*num_silence)

            precursor_list[i] = scale_rms(curr_precursor, rms_target)
        return(precursor_list)



    low = np.arange(800.0, 1800.0+100.0, 100.0) # mean 1300
    mid = np.arange(1800.0, 2800.0+100.0, 100.0) # mean 2300
    high = np.arange(2800.0, 3800.0+100.0, 100.0) # mean 3300

    lowmid = [create_precursor_freqs(low, mid) for i in range(num_trials_per_cond*len(stim_files))]
    midlow = [create_precursor_freqs(mid, low) for i in range(num_trials_per_cond*len(stim_files))]
    highmid = [create_precursor_freqs(high, mid) for i in range(num_trials_per_cond*len(stim_files))]
    midhigh = [create_precursor_freqs(mid, high) for i in range(num_trials_per_cond*len(stim_files))]

    holt_low_mean = np.arange(1300.0, 2300.0+50.0, 50.0) # mean 1800
    holt_high_mean = np.arange(2300.0, 3300.0+50.0, 50.0) # mean 2800
    holt_low_variance = np.arange(1800.0, 2800.0+50.0, 50.0) # mean 2300
    holt_high_variance = np.arange(1300.0, 3300.0+50.0, 50.0) # mean 2300 

    lowmean = [create_precursor_freqs(holt_low_mean) for i in range(num_trials_per_cond*len(stim_files))]
    highmean = [create_precursor_freqs(holt_high_mean) for i in range(num_trials_per_cond*len(stim_files))]
    lowvariance = [create_precursor_freqs(holt_low_variance) for i in range(num_trials_per_cond*len(stim_files))]
    highvariance = [create_precursor_freqs(holt_high_variance) for i in range(num_trials_per_cond*len(stim_files))]

    #lowmid = [create_precursor_freqs(low, mid)]


    #num_precursors = 5
    num_precursors = 20
    # This is taking just just the first precursor. In the future maybe save the log melspec for a bunch of precursors?


    ### HOLT EXPERIMENTS ###

    ## Experiment 1
    low_mean_precursors = create_precursors(lowmean, sr, tone_dur)
    high_mean_precursors = create_precursors(highmean, sr, tone_dur)
    low_variance_precursors = create_precursors(lowvariance, sr, tone_dur)
    high_variance_precursors = create_precursors(highvariance, sr, tone_dur)

    for i in range(num_precursors):
        low_mean_melspec = librosa.feature.melspectrogram(y=low_mean_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        low_mean_logmelspec = np.log(low_mean_melspec + eps)

        high_mean_melspec = librosa.feature.melspectrogram(y=high_mean_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        high_mean_logmelspec = np.log(high_mean_melspec + eps)

        low_variance_melspec = librosa.feature.melspectrogram(y=low_variance_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        low_variance_logmelspec = np.log(low_variance_melspec + eps)

        high_variance_melspec = librosa.feature.melspectrogram(y=high_variance_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        high_variance_logmelspec = np.log(high_variance_melspec + eps)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_low_mean%s.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(low_mean_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_high_mean%s.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(high_mean_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_low_variance%s.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(low_variance_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_high_variance%s.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(high_variance_logmelspec, f)


    # Experiment 2 (Long silence: 1300 ms)

    low_mean_precursors = create_precursors(lowmean, sr, tone_dur, num_silence = 26)
    high_mean_precursors = create_precursors(highmean, sr, tone_dur, num_silence = 26)
    low_variance_precursors = create_precursors(lowvariance, sr, tone_dur, num_silence = 26)
    high_variance_precursors = create_precursors(highvariance, sr, tone_dur, num_silence = 26)

    for i in range(num_precursors):
        low_mean_melspec = librosa.feature.melspectrogram(y=low_mean_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        low_mean_logmelspec = np.log(low_mean_melspec + eps)

        high_mean_melspec = librosa.feature.melspectrogram(y=high_mean_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        high_mean_logmelspec = np.log(high_mean_melspec + eps)

        low_variance_melspec = librosa.feature.melspectrogram(y=low_variance_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        low_variance_logmelspec = np.log(low_variance_melspec + eps)

        high_variance_melspec = librosa.feature.melspectrogram(y=high_variance_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        high_variance_logmelspec = np.log(high_variance_melspec + eps)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_low_mean%s_rep_silence26.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(low_mean_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_high_mean%s_rep_silence26.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(high_mean_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_low_variance%s_rep_silence26.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(low_variance_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_high_variance%s_rep_silence26.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(high_variance_logmelspec, f)


    # Experiment 2 (Short silence: 500 ms)

    low_mean_precursors = create_precursors(lowmean, sr, tone_dur, num_silence = 10)
    high_mean_precursors = create_precursors(highmean, sr, tone_dur, num_silence = 10)
    low_variance_precursors = create_precursors(lowvariance, sr, tone_dur, num_silence = 10)
    high_variance_precursors = create_precursors(highvariance, sr, tone_dur, num_silence = 10)

    for i in range(num_precursors):
        low_mean_melspec = librosa.feature.melspectrogram(y=low_mean_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        low_mean_logmelspec = np.log(low_mean_melspec + eps)

        high_mean_melspec = librosa.feature.melspectrogram(y=high_mean_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        high_mean_logmelspec = np.log(high_mean_melspec + eps)

        low_variance_melspec = librosa.feature.melspectrogram(y=low_variance_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        low_variance_logmelspec = np.log(low_variance_melspec + eps)

        high_variance_melspec = librosa.feature.melspectrogram(y=high_variance_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        high_variance_logmelspec = np.log(high_variance_melspec + eps)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_low_mean%s_rep_silence10.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(low_mean_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_high_mean%s_rep_silence10.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(high_mean_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_low_variance%s_rep_silence10.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(low_variance_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_high_variance%s_rep_silence10.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(high_variance_logmelspec, f)


   # Experiment 3 (Repeated standard 13 times)

    low_mean_precursors = create_precursors(lowmean, sr, tone_dur, num_standard = 13)
    high_mean_precursors = create_precursors(highmean, sr, tone_dur, num_standard = 13)
    low_variance_precursors = create_precursors(lowvariance, sr, tone_dur, num_standard = 13)
    high_variance_precursors = create_precursors(highvariance, sr, tone_dur, num_standard = 13)

    for i in range(num_precursors):
        low_mean_melspec = librosa.feature.melspectrogram(y=low_mean_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        low_mean_logmelspec = np.log(low_mean_melspec + eps)

        high_mean_melspec = librosa.feature.melspectrogram(y=high_mean_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        high_mean_logmelspec = np.log(high_mean_melspec + eps)

        low_variance_melspec = librosa.feature.melspectrogram(y=low_variance_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        low_variance_logmelspec = np.log(low_variance_melspec + eps)

        high_variance_melspec = librosa.feature.melspectrogram(y=high_variance_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        high_variance_logmelspec = np.log(high_variance_melspec + eps)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_low_mean%s_rep_standard13.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(low_mean_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_high_mean%s_rep_standard13.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(high_mean_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_low_variance%s_rep_standard13.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(low_variance_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_high_variance%s_rep_standard13.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(high_variance_logmelspec, f)


   # Experiment 3 (Repeated standard 5 times)

    low_mean_precursors = create_precursors(lowmean, sr, tone_dur, num_standard = 5)
    high_mean_precursors = create_precursors(highmean, sr, tone_dur, num_standard = 5)
    low_variance_precursors = create_precursors(lowvariance, sr, tone_dur, num_standard = 5)
    high_variance_precursors = create_precursors(highvariance, sr, tone_dur, num_standard = 5)

    for i in range(num_precursors):
        low_mean_melspec = librosa.feature.melspectrogram(y=low_mean_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        low_mean_logmelspec = np.log(low_mean_melspec + eps)


        high_mean_melspec = librosa.feature.melspectrogram(y=high_mean_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        high_mean_logmelspec = np.log(high_mean_melspec + eps)

        low_variance_melspec = librosa.feature.melspectrogram(y=low_variance_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        low_variance_logmelspec = np.log(low_variance_melspec + eps)

        high_variance_melspec = librosa.feature.melspectrogram(y=high_variance_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
        high_variance_logmelspec = np.log(high_variance_melspec + eps)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_low_mean%s_rep_standard5.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(low_mean_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_high_mean%s_rep_standard5.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(high_mean_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_low_variance%s_rep_standard5.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(low_variance_logmelspec, f)

        pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'holt_high_variance%s_rep_standard5.pkl'%i
        with open(pklfile, 'wb') as f:
            pickle.dump(high_variance_logmelspec, f)

### END ###

    # # Experiment 3 (Repeated standard 13 times)

    # for i in range(num_precursors):
    #     lowmid_precursors = create_precursors(lowmid, sr, tone_dur, num_standard = 13, num_standard_isi = 13)
    #     lowmid_melspec = librosa.feature.melspectrogram(y=lowmid_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    #     lowmid_logmelspec = np.log(lowmid_melspec + eps)

    #     midlow_precursors = create_precursors(midlow, sr, tone_dur, num_standard = 13, num_standard_isi = 13)
    #     midlow_melspec = librosa.feature.melspectrogram(y=midlow_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    #     midlow_logmelspec = np.log(midlow_melspec + eps)

    #     highmid_precursors = create_precursors(highmid, sr, tone_dur, num_standard = 13, num_standard_isi = 13)
    #     highmid_melspec = librosa.feature.melspectrogram(y=highmid_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    #     highmid_logmelspec = np.log(highmid_melspec + eps)

    #     midhigh_precursors = create_precursors(midhigh, sr, tone_dur, num_standard = 13, num_standard_isi = 13)
    #     midhigh_melspec = librosa.feature.melspectrogram(y=midhigh_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    #     midhigh_logmelspec = np.log(midhigh_melspec + eps)


    #     pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'lowmid%s_rep_standard13.pkl'%i
    #     with open(pklfile, 'wb') as f:
    #         pickle.dump(lowmid_logmelspec, f)

    #     pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'midlow%s_rep_standard13.pkl'%i
    #     with open(pklfile, 'wb') as f:
    #         pickle.dump(midlow_logmelspec, f)

    #     pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'highmid%s_rep_standard13.pkl'%i
    #     with open(pklfile, 'wb') as f:
    #         pickle.dump(highmid_logmelspec, f)

    #     pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'midhigh%s_rep_standard13.pkl'%i
    #     with open(pklfile, 'wb') as f:
    #         pickle.dump(midhigh_logmelspec, f)


    # # Experiment 3 (Repeated standard 5 times)

    # for i in range(num_precursors):
    #     lowmid_precursors = create_precursors(lowmid, sr, tone_dur, num_standard = 5, num_standard_isi = 13)
    #     lowmid_melspec = librosa.feature.melspectrogram(y=lowmid_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    #     lowmid_logmelspec = np.log(lowmid_melspec + eps)

    #     midlow_precursors = create_precursors(midlow, sr, tone_dur, num_standard = 5, num_standard_isi = 13)
    #     midlow_melspec = librosa.feature.melspectrogram(y=midlow_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    #     midlow_logmelspec = np.log(midlow_melspec + eps)

    #     highmid_precursors = create_precursors(highmid, sr, tone_dur, num_standard = 5, num_standard_isi = 13)
    #     highmid_melspec = librosa.feature.melspectrogram(y=highmid_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    #     highmid_logmelspec = np.log(highmid_melspec + eps)

    #     midhigh_precursors = create_precursors(midhigh, sr, tone_dur, num_standard = 5, num_standard_isi = 13)
    #     midhigh_melspec = librosa.feature.melspectrogram(y=midhigh_precursors[i], sr=sr_target, S=None, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=n_mels)
    #     midhigh_logmelspec = np.log(midhigh_melspec + eps)


    #     pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'lowmid%s_rep_standard5.pkl'%i
    #     with open(pklfile, 'wb') as f:
    #         pickle.dump(lowmid_logmelspec, f)

    #     pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'midlow%s_rep_standard5.pkl'%i
    #     with open(pklfile, 'wb') as f:
    #         pickle.dump(midlow_logmelspec, f)

    #     pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'highmid%s_rep_standard5.pkl'%i
    #     with open(pklfile, 'wb') as f:
    #         pickle.dump(highmid_logmelspec, f)

    #     pklfile = './precursors/%s/%smels/'%(padding,n_mels) + 'midhigh%s_rep_standard5.pkl'%i
    #     with open(pklfile, 'wb') as f:
    #         pickle.dump(midhigh_logmelspec, f)


