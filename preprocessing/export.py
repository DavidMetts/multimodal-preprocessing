"""Stable three-table handoff and JSON replay settings."""
import io
import json
import zipfile
import platform
from importlib.metadata import version
import pandas as pd
from . import text, image, audio
from .inputs import digest

MODALITIES={'text':'text.txt','image':'image.png','audio':'audio.wav'}

def prepare(events,model):
    tables={}; details={}; hashes=[]
    for modality,filename in MODALITIES.items():
        rows=[]; details[modality]={}
        for e in events:
            row=dict(event_id=e.event_id,event_time=e.event_time,source_reference=e.reference(filename),
                     source_sha256=digest(e.files[filename]) if filename in e.files else '',status='missing',error='')
            if filename in e.files:
                try:
                    data=e.files[filename]
                    if modality=='text':
                        if model is None: raise ValueError('Fit or upload a text reference first.')
                        document=data.decode('utf-8-sig'); result=text.transform(document,model)
                        if not result['token_count']: raise ValueError('No word tokens in document.')
                        for index,term in enumerate(model['vocabulary']):
                            for kind in ('count','tf','tfidf','normalized'):
                                row[f'{kind}__{term}']=result[kind][index]
                        row['token_count']=result['token_count']; row['unknown_token_count']=result['unknown_token_count']
                        details[modality][e.event_id]=(document,result)
                    elif modality=='image':
                        result=image.extract(data); row.update(result[0]); details[modality][e.event_id]=result
                    else:
                        result=audio.extract(data); row.update(result[0]); details[modality][e.event_id]=result
                    row['status']='ok'
                    if modality=='audio' and not (result[2].status=='valid').all():
                        row['status']='partial/invalid'; row['error']='At least one second contains invalid or silent frames; measurements left blank.'
                except Exception as exc:
                    row['status']='invalid'; row['error']=str(exc)
            rows.append(row)
        tables[modality]=pd.DataFrame(rows)
    for e in events:
        hashes.append(dict(event_id=e.event_id,event_time=e.event_time,notes=e.notes,
                           sources={name:dict(reference=e.reference(name),sha256=digest(data)) for name,data in e.files.items()}))
    image_features=['mean_red_normalized','mean_green_normalized','mean_blue_normalized','area_fraction_normalized']
    audio_features=[f'second_{i:02}_{suffix}_normalized' for i in range(1,11) for suffix in ('mean_pitch_hz','pitch_variance_hz_squared')]
    text_features=['normalized__'+term for term in model['vocabulary']] if model else []
    # Keep a stable numeric schema even when every event is missing/invalid for a modality.
    expected={
        'text':[f'{kind}__{term}' for term in (model['vocabulary'] if model else []) for kind in ('count','tf','tfidf','normalized')]+['token_count','unknown_token_count'],
        'image':['mean_red','mean_green','mean_blue','selected_pixel_count','canvas_pixel_count','area_fraction']+image_features,
        'audio':[x.removesuffix('_normalized') for x in audio_features]+audio_features}
    for modality,cols in expected.items():
        base=['event_id','event_time','source_reference','source_sha256','status','error']
        tables[modality]=tables[modality].reindex(columns=base+cols)
    metadata=dict(schema_version=1,app_version='1.0.0',events=hashes,text_reference=model,
                  normalized_feature_order=dict(text=text_features,image=image_features,audio=audio_features),
                  image=dict(mask_rule=image.RULE,units='RGB intensity 0..255; area fraction 0..1; pixel counts',normalization='RGB / 255; area unchanged; fixed constants, no collection fitting'),
                  audio=audio.SETTINGS, audio_units='mean: Hz; population variance: Hz squared',
                  missing_values='Blank CSV numeric cells mean missing/invalid, never imputed zero.',
                  text_idf_formula='ln((1 + reference_count)/(1 + document_frequency)) + 1',
                  feature_note='Unequal feature counts; no claim of equivalent information or importance.',
                  runtime=dict(python=platform.python_version(),**{p:version(p) for p in ('numpy','pandas','Pillow','scipy','streamlit')}))
    return tables,details,metadata

def csv_bytes(table):
    return table.to_csv(index=False,lineterminator='\n',float_format='%.12g').encode('utf-8')

def bundle(tables,details,metadata):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for modality,table in tables.items(): z.writestr(modality+'_attributes.csv',csv_bytes(table))
        z.writestr('metadata.json',json.dumps(metadata,indent=2,allow_nan=False))
        if metadata['text_reference']:
            z.writestr('text_reference.json',json.dumps(metadata['text_reference'],indent=2))
        frames=[]
        for event_id,result in details['audio'].items():
            f=result[1].copy(); f.insert(0,'event_id',event_id); frames.append(f)
        if frames: z.writestr('inspection/audio_frames.csv',csv_bytes(pd.concat(frames)))
    return buffer.getvalue()
