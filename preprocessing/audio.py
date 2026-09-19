"""Rising zero crossings with linear interpolation, restricted to steady sine tones."""
from io import BytesIO
import wave
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

SETTINGS=dict(frame_seconds=.1, hop_seconds=.1, min_hz=100., max_hz=1200.,
              silence_rms=1e-4, duration_seconds=10., mean_scale_hz=1000.,
              variance_scale_hz_squared=1000000., variance_ddof=0,
              estimator='mean rising zero-crossing period, linearly interpolated; >=3 crossings; period CV <=0.05',
              normalization='mean Hz / 1000; variance Hz^2 / 1000000; no clipping or per-clip fitting')

def extract(data):
    with wave.open(BytesIO(data),'rb') as w:
        sr=w.getframerate()
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':
            raise ValueError('Expected uncompressed mono 16-bit PCM WAV.')
        if not 8000<=sr<=48000 or w.getnframes()!=10*sr:
            raise ValueError('Expected exactly ten seconds, sample rate 8,000–48,000 Hz.')
        raw=w.readframes(w.getnframes())
        if len(raw)!=20*sr: raise ValueError('Truncated WAV data.')
        signal=np.frombuffer(raw,dtype='<i2').astype(float)/32768
    frame_n=round(sr*.1); rows=[]
    for start in range(0,len(signal)-frame_n+1,frame_n):
        x=signal[start:start+frame_n]; rms=float(np.sqrt(np.mean(x*x))); pitch=np.nan
        status='silent' if rms<SETTINGS['silence_rms'] else 'invalid tone'
        indices=np.where((x[:-1]<=0)&(x[1:]>0))[0]
        if rms>=SETTINGS['silence_rms'] and len(indices)>=3:
            crossings=indices-x[indices]/(x[indices+1]-x[indices]); periods=np.diff(crossings)
            candidate=sr/periods.mean()
            if 100<=candidate<=1200 and periods.std()/periods.mean()<=.05:
                pitch=float(candidate); status='valid'
        rows.append(dict(start_seconds=start/sr,center_seconds=(start+frame_n/2)/sr,
                         end_seconds=(start+frame_n)/sr,pitch_hz=pitch,rms=rms,status=status))
    frames=pd.DataFrame(rows); summaries=[]; features={}
    for second in range(10):
        part=frames[(frames.center_seconds>=second)&(frames.center_seconds<second+1)]
        valid=part.pitch_hz.dropna(); count=len(valid)
        # Require all ten frames: do not conceal gaps with partial summaries.
        mean=float(valid.mean()) if count==10 else np.nan
        variance=float(valid.var(ddof=0)) if count==10 else np.nan
        summaries.append(dict(second=second+1,start_seconds=second,end_seconds=second+1,
                              valid_frames=count,total_frames=len(part),mean_pitch_hz=mean,
                              pitch_variance_hz_squared=variance,status='valid' if count==10 else 'invalid/silent frames'))
        prefix=f'second_{second+1:02}'
        features[prefix+'_mean_pitch_hz']=mean
        features[prefix+'_pitch_variance_hz_squared']=variance
    raw_keys=list(features)
    for key in raw_keys:
        features[key+'_normalized']=features[key]/(1000000 if 'variance' in key else 1000)
    f,t,s=spectrogram(signal,fs=sr,nperseg=512,noverlap=384)
    keep=f<=1500
    return features,frames,pd.DataFrame(summaries),(f[keep],t,10*np.log10(np.maximum(s[keep],1e-12)))
