"""Bounded in-memory imports. Archive paths are never extracted to disk."""
from dataclasses import dataclass, field
from pathlib import PurePosixPath, Path
from io import BytesIO
import hashlib
import json
import zipfile

MAX_BYTES=25*1024*1024
MAX_EVENTS=50
NAMES={'text.txt','image.png','audio.wav','manifest.json'}

@dataclass
class Event:
    event_id: str
    event_time: str
    files: dict
    source_prefix: str
    notes: list = field(default_factory=list)

    def reference(self,name):
        return self.source_prefix+'/'+name if name in self.files else ''

def digest(data):
    return hashlib.sha256(data).hexdigest()

def make_event(files,prefix):
    manifest={}; notes=[]
    if 'manifest.json' in files:
        try:
            manifest=json.loads(files['manifest.json'].decode('utf-8-sig'))
            if not isinstance(manifest,dict): raise ValueError()
        except (ValueError,UnicodeError):
            raise ValueError('Invalid manifest JSON in '+prefix)
    event_id=manifest.get('event_id')
    if event_id is not None and (not isinstance(event_id,str) or not event_id.strip() or len(event_id)>200):
        raise ValueError('Invalid event ID in '+prefix)
    if not event_id:
        event_id='unlinked_'+digest(b''.join(name.encode()+files[name] for name in sorted(files)))[:16]
        notes.append('No manifest event ID: content-derived ID; original alignment/time cannot be established.')
    event_time=manifest.get('event_time','')
    if not isinstance(event_time,str) or len(event_time)>100: raise ValueError('Invalid event time.')
    # Deliberately do not read flavor, scoop, tone, frequency, or source filenames from metadata.
    return Event(event_id,event_time,files,prefix,notes)

def import_zips(uploads):
    if sum(len(data) for _,data in uploads)>MAX_BYTES:
        raise ValueError('Uploads exceed the combined 25 MB limit.')
    events=[]; expanded=0
    for filename,data in uploads:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            infos=archive.infolist()
            if len(infos)>250: raise ValueError('ZIP has more than 250 entries.')
            groups={}; seen=set()
            for info in infos:
                # ZipInfo normalizes Windows backslashes on Windows; inspect original too.
                name=info.orig_filename; path=PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                    raise ValueError('Unsafe ZIP path rejected: '+name)
                if info.is_dir(): continue
                if name in seen: raise ValueError('Duplicate ZIP member: '+name)
                seen.add(name)
                if path.name not in NAMES: raise ValueError('Unexpected archive file: '+name)
                if (info.external_attr >> 16)&0o170000 == 0o120000: raise ValueError('Archive links are unsupported.')
                expanded+=info.file_size
                if expanded>MAX_BYTES or info.file_size>2*1024*1024:
                    raise ValueError('Uncompressed collection exceeds limits (25 MB total, 2 MB per file).')
                groups.setdefault(str(path.parent),{})[path.name]=archive.read(info)
            for folder,files in sorted(groups.items()):
                events.append(make_event(files,filename+':'+folder))
    return validate_events(events)

def validate_events(events):
    if len(events)>MAX_EVENTS: raise ValueError('Maximum 50 events per collection.')
    ids=[e.event_id for e in events]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate event IDs: remove duplicate events before importing.')
    return events

def folder_zip(folder):
    """Local CLI helper only; hosted UI intentionally cannot browse server folders."""
    root=Path(folder).resolve(strict=True); buffer=BytesIO(); total=0; count=0
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_symlink(): raise ValueError('Symbolic links are unsupported.')
            if path.is_file() and path.name in NAMES:
                count+=1; total+=path.stat().st_size
                if count>250 or total>MAX_BYTES or path.stat().st_size>2*1024*1024:
                    raise ValueError('Folder exceeds collection limits.')
                archive.write(path,str(path.relative_to(root)).replace('\\','/'))
    result=buffer.getvalue(); import_zips([('collection.zip',result)])
    return result
