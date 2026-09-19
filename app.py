"""Streamlit UI; modality algorithms and export logic live in preprocessing/."""
import io
import json
from pathlib import Path
import zipfile
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing import text
from preprocessing.inputs import import_zips, make_event, validate_events, MAX_BYTES, digest
from preprocessing.export import prepare, csv_bytes, bundle

st.set_page_config(page_title='Three Source Lab · Preprocessing',page_icon='🔬',layout='wide')
st.caption('THREE SOURCE LAB / STAGE 02')
st.title('From raw sources to measurable attributes')
st.write('Inspect text, ice-cream images, and steady tones. Export three linked tables for the next stage.')
st.caption('Preprocessing only • Raw sources preserved • No labels used as features')

with st.sidebar:
    st.header('1 · Load a collection')
    mode=st.radio('Input route',['Demo collection','Event ZIP uploads','Single event files'])
    events=[]
    try:
        if mode=='Demo collection':
            events=import_zips([('demo_collection.zip',(Path(__file__).parent/'demo_collection.zip').read_bytes())])
            st.caption('Nine events copied from the original generator assets. No generated replacements.')
            st.download_button('Download raw demo collection',(Path(__file__).parent/'demo_collection.zip').read_bytes(),'demo_collection.zip')
        elif mode=='Event ZIP uploads':
            uploads=st.file_uploader('Upload event ZIPs or a collection ZIP',type=['zip'],accept_multiple_files=True)
            if uploads: events=import_zips([(u.name,u.getvalue()) for u in uploads])
            st.caption('Up to 50 events, 25 MB combined expanded data, 2 MB per raw file. Each event folder contains image.png, audio.wav, text.txt and manifest.json.')
        else:
            st.caption('Add one event at a time. A manifest preserves its original ID and timestamp. Without it, a content-derived ID marks the sources as unlinked.')
            files={}
            for name,kind in [('text.txt','txt'),('image.png','png'),('audio.wav','wav'),('manifest.json','json')]:
                u=st.file_uploader(name,type=[kind])
                if u:
                    if u.size>2*1024*1024: raise ValueError('Each source must be at most 2 MB.')
                    files[name]=u.getvalue()
            if st.button('Add event to collection',disabled=not files):
                prior=st.session_state.get('single_events',[])
                if sum(len(d) for e in prior for d in e.files.values())+sum(map(len,files.values()))>MAX_BYTES:
                    raise ValueError('Collection exceeds 25 MB.')
                candidate=make_event(files,'single-upload-'+digest(b''.join(files.values()))[:12])
                st.session_state.single_events=validate_events(prior+[candidate])
            if st.button('Clear single-event collection'): st.session_state.single_events=[]
            events=st.session_state.get('single_events',[])
    except Exception as exc:
        st.error(str(exc)); events=[]
    with st.expander('Import a local folder'):
        st.write('A hosted app cannot read folders on your computer. Zip event folders together, or run this helper locally and upload its result:')
        st.code('python package_folder.py "path/to/events" collection.zip')
    st.divider()
    st.header('2 · Text reference')
    st.caption('Choose documents, then freeze their shared vocabulary and IDF. Later imports keep the same reference until you explicitly refit.')
    valid_documents={}
    for e in events:
        if 'text.txt' in e.files:
            try:
                doc=e.files['text.txt'].decode('utf-8-sig')
                if text.tokens(doc): valid_documents[e.event_id]=doc
            except UnicodeError: pass
    reference_ids=st.multiselect('Reference documents',list(valid_documents),default=list(valid_documents))
    if st.button('Fit and freeze reference',disabled=not reference_ids):
        try:
            refs=[dict(event_id=e.event_id,source_reference=e.reference('text.txt'),sha256=digest(e.files['text.txt'])) for e in events if e.event_id in reference_ids]
            st.session_state.model=text.fit([valid_documents[r] for r in reference_ids],refs)
        except ValueError as exc: st.error(str(exc))
    frozen=st.file_uploader('Restore text_reference.json',type=['json'])
    if st.button('Use uploaded reference',disabled=frozen is None):
        try:
            if frozen.size>1024*1024: raise ValueError('Reference exceeds 1 MB.')
            st.session_state.model=text.validate(json.loads(frozen.getvalue()))
        except Exception as exc: st.error('Cannot restore reference: '+str(exc))
    if 'model' not in st.session_state and valid_documents:
        refs=[dict(event_id=e.event_id,source_reference=e.reference('text.txt'),sha256=digest(e.files['text.txt'])) for e in events if e.event_id in valid_documents]
        try: st.session_state.model=text.fit(list(valid_documents.values()),refs)
        except ValueError as exc: st.error(str(exc))
    model=st.session_state.get('model')
    if model:
        st.success(f"Frozen: {model['reference_count']} documents · {len(model['vocabulary'])} terms")
        st.download_button('Download frozen text reference',json.dumps(model,indent=2),'text_reference.json','application/json')
        with st.expander('Reference provenance'):
            st.json(model['reference_sources'])

if not events:
    st.info('Choose the demo or upload sources to begin. Missing modalities remain missing.')
    st.stop()

tables,details,metadata=prepare(events,model)
metrics=st.columns(4)
metrics[0].metric('Events',len(events))
metrics[1].metric('Text attributes',len(model['vocabulary']) if model else 0)
metrics[2].metric('Image attributes',4)
metrics[3].metric('Audio attributes',20)
st.caption('Attribute counts describe the normalized candidates; counts do not establish importance or equal information.')
with st.expander('Collection quality and source links',expanded=False):
    for modality,table in tables.items():
        st.write(modality.title()); st.dataframe(table.iloc[:,:6],hide_index=True,width='stretch')
    for e in events:
        for note in e.notes: st.warning(e.event_id+': '+note)
event_id=st.selectbox('Event to inspect',[e.event_id for e in events]); event=next(e for e in events if e.event_id==event_id)
st.caption('Event time: '+(event.event_time or 'not available'))

def missing(modality):
    row=tables[modality].set_index('event_id').loc[event_id]
    st.warning(f"{modality.title()}: {row['status']}. {row['error']}")

text_tab,image_tab,audio_tab=st.tabs(['Text','Image','Audio'])
with text_tab:
    st.subheader('Words → shared TF-IDF coordinates')
    st.caption("Lowercase English word tokens; internal apostrophes retained. No stop words, stemming, labels, or flavor weighting. Vanilla, chocolate and strawberry are ordinary words.")
    if model and model['reference_count']==1: st.info('Only one reference document: cross-document distinctiveness cannot yet be established; every observed term has IDF = 1.')
    if event_id in details['text']:
        doc,result=details['text'][event_id]; st.text_area('Original text',doc,height=130,disabled=True)
        if result['unknown_token_count']: st.warning(f"{result['unknown_token_count']} tokens fall outside the frozen vocabulary. They are counted in document length but get no coordinate.")
        word_counts=pd.Series(text.tokens(doc)).value_counts().rename_axis('word').reset_index(name='count')
        st.altair_chart(alt.Chart(word_counts).mark_bar(color='#287A78').encode(x=alt.X('word:N',sort='-y'),y='count:Q',tooltip=['word','count']).properties(height=240),width='stretch')
        words=pd.DataFrame(dict(term=model['vocabulary'],count=result['count'],term_frequency=result['tf'],document_frequency=model['document_frequency'],inverse_document_frequency=model['idf'],tf_idf=result['tfidf'],normalized_tf_idf=result['normalized']))
        st.dataframe(words,hide_index=True,width='stretch')
    else: missing('text')
    st.markdown('**Collection comparison · raw TF-IDF**')
    if model and details['text']:
        heat=pd.DataFrame([{'event':e,'term':t,'TF-IDF':float(r[1]['tfidf'][i])} for e,r in details['text'].items() for i,t in enumerate(model['vocabulary'])])
        st.altair_chart(alt.Chart(heat).mark_rect(aria=False).encode(x=alt.X('term:N',sort=model['vocabulary']),y='event:N',color=alt.Color('TF-IDF:Q',scale=alt.Scale(scheme='tealblues')),tooltip=['event','term','TF-IDF']).properties(height=max(120,24*len(details['text']))),width='stretch')
    st.caption('TF = count / all tokens. IDF = ln((1 + reference documents)/(1 + document frequency)) + 1. Export includes counts, TF, TF-IDF and its L2-normalized vector; DF/IDF live in the frozen reference.')
with image_tab:
    st.subheader('Ice-cream pixels → color and canvas fraction')
    if event_id in details['image']:
        measurements,overlay,mask=details['image'][event_id]; cols=st.columns(2)
        cols[0].image(event.files['image.png'],caption='Original PNG',width=310)
        cols[1].image(overlay,caption='Selected pixels in full color; excluded pixels dimmed',width=310)
        st.dataframe(pd.DataFrame([measurements]),hide_index=True,width='stretch')
    else: missing('image')
    st.caption('Fixed 512 × 640 controlled canvas only. Geometric scoop region excludes exact background, cone and outline colors. No flavor lookup. RGB / 255; selected pixels / full canvas. Resized, shaded, compressed or redesigned artwork needs a new mask.')
with audio_tab:
    st.subheader('Ten seconds → ten ordered mean / variance pairs')
    if 'audio.wav' in event.files: st.audio(event.files['audio.wav'],format='audio/wav',autoplay=False)
    if event_id in details['audio']:
        features,frames,seconds,(frequencies,times,power)=details['audio'][event_id]
        # Downsample display time only; pitch uses the original waveform.
        stride=max(1,int(np.ceil(len(times)/100))); shown=times[::stride]
        dt=(times[1]-times[0])*stride; df=frequencies[1]-frequencies[0]
        grid=pd.DataFrame({'time':np.tile(shown,len(frequencies)),'frequency':np.repeat(frequencies,len(shown)),'power_db':power[:,::stride].ravel()})
        grid['time_end']=np.minimum(grid.time+dt,10); grid['frequency_end']=grid.frequency+df
        st.altair_chart(alt.Chart(grid).mark_rect(aria=False).encode(x=alt.X('time:Q',scale=alt.Scale(domain=[0,10]),title='Time (seconds)'),x2='time_end:Q',y=alt.Y('frequency:Q',title='Frequency (Hz)'),y2='frequency_end:Q',color=alt.Color('power_db:Q',title='PSD dB'),tooltip=['time','frequency','power_db']).properties(height=220),width='stretch')
        st.caption('Spectrogram is a display; pitch is measured from waveform crossings, not spectrogram cell averages.')
        means=alt.Chart(seconds).mark_line(point=True,color='#287A78').encode(x=alt.X('second:Q',scale=alt.Scale(domain=[1,10]),axis=alt.Axis(tickMinStep=1)),y=alt.Y('mean_pitch_hz:Q',scale=alt.Scale(domain=[0,1200])),tooltip=['second','mean_pitch_hz'])
        variances=alt.Chart(seconds).mark_line(point=True,color='#AC6443').encode(x=alt.X('second:Q',scale=alt.Scale(domain=[1,10]),axis=alt.Axis(tickMinStep=1)),y=alt.Y('pitch_variance_hz_squared:Q',scale=alt.Scale(domain=[0,1])),tooltip=['second','pitch_variance_hz_squared'])
        st.altair_chart(alt.vconcat(means.properties(height=160),variances.properties(height=160)).resolve_scale(x='shared'),width='stretch')
        st.dataframe(seconds,hide_index=True,width='stretch')
        with st.expander('Inspect 100 timestamped analysis frames'):
            st.dataframe(frames,hide_index=True,width='stretch')
            st.download_button('Download frame measurements',csv_bytes(frames),'audio_frames.csv','text/csv')
        if not (seconds.status=='valid').all(): st.warning('Invalid/silent windows remain blank. All ten valid frames are required for a one-second summary.')
    else: missing('audio')
    st.caption('100 ms non-overlapping frames; rising zero crossings with linear interpolation. Restricted to clean single tones, 100–1200 Hz. Mean / 1000 Hz; population variance / 1,000,000 Hz². Fixed scales preserve low/middle/high differences without amplifying tiny variance. One-second summaries lose within-second order; frame times are retained in the bundle.')

st.divider(); st.subheader('3 · Export the preprocessing handoff')
st.caption('Three separate attribute tables, linked by event ID and source hash. Blank numeric cells mean missing or invalid. Keep your raw ZIPs alongside these outputs.')
cols=st.columns(4)
for index,(modality,table) in enumerate(tables.items()):
    cols[index].download_button(modality.title()+' CSV',csv_bytes(table),modality+'_attributes.csv','text/csv',width='stretch')
cols[3].download_button('All tables + metadata',bundle(tables,details,metadata),'preprocessed_attributes.zip','application/zip',width='stretch')
with st.expander('Inspect export metadata'):
    st.json(metadata)
with st.expander('Original files for selected event'):
    for name,data in event.files.items(): st.download_button('Download '+name,data,name,key='raw_'+name)
