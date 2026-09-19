import io,json,unittest,zipfile,wave
from pathlib import Path
import numpy as np
import pandas as pd
from preprocessing import text,image,audio
from preprocessing.inputs import import_zips,make_event
from preprocessing.export import prepare,bundle

ROOT=Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events=import_zips([('demo.zip',(ROOT/'demo_collection.zip').read_bytes())])
        cls.truth=json.loads((ROOT/'tests/fixture_truth.json').read_text())
        cls.model=text.fit([e.files['text.txt'].decode() for e in cls.events],[e.event_id for e in cls.events])

    def test_text_shared_reference(self):
        model=text.validate(self.model)
        for event,truth in zip(self.events,self.truth):
            r=text.transform(event.files['text.txt'].decode(),model)
            for flavor in ('vanilla','chocolate','strawberry'):
                i=model['vocabulary'].index(flavor)
                self.assertEqual(r['count'][i],5 if flavor==truth['fixture_truth']['flavor'] else 0)
                self.assertEqual(model['document_frequency'][i],3)
            self.assertAlmostEqual(np.linalg.norm(r['normalized']),1)
            np.testing.assert_array_equal(r['tfidf'],text.transform(event.files['text.txt'].decode(),json.loads(json.dumps(model)))['tfidf'])
        one=text.fit(['vanilla vanilla cream'],['one'])
        self.assertEqual(one['idf'],[1.,1.])
        self.assertEqual(text.transform('unseen',one)['unknown_token_count'],1)
        bad=one|{'lowercase':False}
        with self.assertRaises(ValueError): text.validate(bad)

    def test_image_invariants(self):
        colors={'vanilla':[255,248,231],'chocolate':[121,80,57],'strawberry':[237,138,168]}
        areas={}
        for e,t in zip(self.events,self.truth):
            result,_,mask=image.extract(e.files['image.png']); truth=t['fixture_truth']
            np.testing.assert_array_equal([result[k] for k in ('mean_red','mean_green','mean_blue')],colors[truth['flavor']])
            areas.setdefault(truth['flavor'],[]).append(result['selected_pixel_count'])
            self.assertAlmostEqual(result['area_fraction'],mask.sum()/(512*640))
            self.assertEqual(result['mean_red_normalized'],colors[truth['flavor']][0]/255)
        for values in areas.values(): self.assertTrue(values[0]<values[1]<values[2])
        self.assertEqual(areas['vanilla'],areas['chocolate']); self.assertEqual(areas['vanilla'],areas['strawberry'])

    def test_audio_order_and_silence(self):
        for e,t in zip(self.events[:3],self.truth[:3]):
            f,frames,seconds,_=audio.extract(e.files['audio.wav'])
            self.assertEqual(len(f),40); self.assertEqual(len(frames),100)
            np.testing.assert_allclose(seconds.mean_pitch_hz,t['fixture_truth']['tone_hz'],atol=.02)
            self.assertLess(seconds.pitch_variance_hz_squared.max(),.001)
            self.assertEqual(list(f)[:2],['second_01_mean_pitch_hz','second_01_pitch_variance_hz_squared'])
            self.assertEqual(list(f)[18:20],['second_10_mean_pitch_hz','second_10_pitch_variance_hz_squared'])
        b=io.BytesIO()
        with wave.open(b,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(bytes(320000))
        f,frames,seconds,_=audio.extract(b.getvalue())
        self.assertTrue(seconds.mean_pitch_hz.isna().all());self.assertTrue(seconds.pitch_variance_hz_squared.isna().all())
        self.assertTrue((frames.status=='silent').all())

    def test_chronological_pitch_changes(self):
        sr=16000; wave_data=[]
        for second in range(10):
            freq=220 if second<5 else 880
            wave_data.append(np.rint(.2*32767*np.sin(2*np.pi*freq*np.arange(sr)/sr)).astype('<i2'))
        b=io.BytesIO()
        with wave.open(b,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(np.concatenate(wave_data).tobytes())
        f,_,_,_=audio.extract(b.getvalue())
        self.assertAlmostEqual(f['second_05_mean_pitch_hz'],220,places=2)
        self.assertAlmostEqual(f['second_06_mean_pitch_hz'],880,places=2)

    def test_no_label_leakage_and_exports(self):
        tables,details,meta=prepare(self.events,self.model)
        changed=[]
        for e in self.events:
            files=e.files.copy();files['manifest.json']=json.dumps(dict(event_id=e.event_id,event_time=e.event_time,sources={'audio':{'frequency_hz':9999},'image':{'flavor':'incorrect','scoops':999}})).encode()
            changed.append(make_event(files,'misleading-flavor-name'))
        t2,_,_=prepare(changed,self.model)
        for modality in tables:
            cols=[c for c in tables[modality] if c!='source_reference']
            pd.testing.assert_frame_equal(tables[modality][cols],t2[modality][cols])
            self.assertEqual(list(tables[modality].event_id),[e.event_id for e in self.events])
        self.assertEqual(len(meta['normalized_feature_order']['audio']),20)
        with zipfile.ZipFile(io.BytesIO(bundle(tables,details,meta))) as z:
            self.assertTrue({'text_attributes.csv','image_attributes.csv','audio_attributes.csv','metadata.json','text_reference.json','inspection/audio_frames.csv'}<=set(z.namelist()))
            text.validate(json.loads(z.read('text_reference.json')))

    def test_missing_invalid_and_archive_safety(self):
        events=[make_event({'text.txt':b'vanilla'},'standalone')]
        t,_,_=prepare(events,self.model)
        self.assertEqual(t['image'].status[0],'missing');self.assertTrue(pd.isna(t['image'].mean_red[0]))
        invalid=[make_event({'audio.wav':b'broken','image.png':b'broken','text.txt':b'\xff'},'invalid')]
        t,_,_=prepare(invalid,self.model)
        self.assertTrue(all(table.status[0]=='invalid' for table in t.values()))
        for name in ('../text.txt','/text.txt','C:/text.txt','a\\text.txt','a/./text.txt'):
            b=io.BytesIO()
            member=zipfile.ZipInfo('placeholder'); member.filename=name
            with zipfile.ZipFile(b,'w') as z:z.writestr(member,b'bad')
            with self.assertRaises(ValueError): import_zips([('bad.zip',b.getvalue())])
        with self.assertRaises(ValueError): import_zips([('one.zip',(ROOT/'demo_collection.zip').read_bytes()),('two.zip',(ROOT/'demo_collection.zip').read_bytes())])
        timestamp='2026-09-19T12:30:00+00:00'
        event=make_event({'text.txt':b'vanilla','manifest.json':json.dumps({'event_id':'snapshot_1','event_time':timestamp}).encode()},'source')
        t,_,meta=prepare([event],self.model)
        self.assertTrue(all(table.event_time[0]==timestamp for table in t.values()))
        self.assertEqual(meta['events'][0]['event_time'],timestamp)

    def test_dashboard_controls(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual([t.label for t in app.tabs],['Text','Image','Audio'])
        self.assertEqual(len(app.get('download_button')),11)
        app.selectbox[0].set_value('demo_09').run(); self.assertFalse(app.exception)
        app.multiselect[0].set_value(['demo_01']).run()
        next(b for b in app.button if b.label=='Fit and freeze reference').click().run()
        self.assertEqual(app.session_state['model']['reference_count'],1)
        self.assertFalse(app.exception)
        app.radio[0].set_value('Event ZIP uploads').run();self.assertFalse(app.exception)
        self.assertTrue(any('upload sources' in i.value for i in app.info))

if __name__=='__main__': unittest.main()
