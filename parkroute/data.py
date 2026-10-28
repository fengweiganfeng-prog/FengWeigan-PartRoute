"""Public attraction data: download, checksum, schema validation and projection."""
from pathlib import Path
import hashlib, io, json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

COMMIT = 'ff457ed9c177c98dfe92ec5d8df608dfd8de0a8a'
BASE = f'https://raw.githubusercontent.com/junhua/SCAIR/{COMMIT}'
URL = BASE + '/data/POI-MagicK.csv'
SELECTED = [1, 3, 7, 9, 12, 13, 15, 19, 21, 24]
REQUIRED = ['poiID','poiName','lat','long','duration','capacity','rating','n_reviews']

def validate(df):
    if list(df.columns) != REQUIRED:
        raise ValueError(f'Unexpected CSV schema: {list(df.columns)}')
    if df.empty or df.isna().any().any():
        raise ValueError('Empty data or missing values')
    if df.poiID.duplicated().any() or df.poiName.duplicated().any():
        raise ValueError('Duplicate attraction identifiers or names')
    for col in REQUIRED[2:] + ['poiID']:
        values = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite values in {col}')
    if not df.lat.between(-90,90).all() or not df.long.between(-180,180).all():
        raise ValueError('Invalid coordinates')
    if not df.rating.between(0,5).all() or (df[['duration','capacity']] <= 0).any().any():
        raise ValueError('Invalid ratings, durations or capacities')
    if (df.n_reviews < 0).any() or (df.poiID % 1 != 0).any() or (df.n_reviews % 1 != 0).any():
        raise ValueError('Invalid integer fields')
    return {'rows':len(df),'columns':len(df.columns),'missing_values':0,'duplicate_ids':0,
            'rating_min':float(df.rating.min()),'rating_max':float(df.rating.max()),
            'duration_min':float(df.duration.min()),'duration_max':float(df.duration.max())}

def download(root, refresh=False):
    root = Path(root)
    raw = root/'data/raw'; raw.mkdir(parents=True,exist_ok=True)
    target = raw/'POI-MagicK.csv'
    manifest = raw/'download_manifest.json'
    if target.exists() and manifest.exists() and not refresh:
        meta = json.loads(manifest.read_text())
        if hashlib.sha256(target.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Cached CSV checksum mismatch')
        validate(pd.read_csv(target))
        return meta
    last = None
    for attempt in range(3):
        try:
            response = requests.get(URL,timeout=(10,30))
            response.raise_for_status()
            content = response.content
            df = pd.read_csv(io.BytesIO(content))
            checks = validate(df)
            target.write_bytes(content)
            meta = {'url':URL,'resolved_url':response.url,'status_code':response.status_code,
                    'content_type':response.headers.get('Content-Type'),'bytes':len(content),
                    'sha256':hashlib.sha256(content).hexdigest(),'commit':COMMIT,
                    'checked_utc':datetime.now(timezone.utc).isoformat(),'validation':checks}
            manifest.write_text(json.dumps(meta,indent=2))
            license_response = requests.get(BASE+'/LICENSE',timeout=(10,30))
            license_response.raise_for_status()
            (raw/'SCAIR_LICENSE.txt').write_text(license_response.text)
            return meta
        except (requests.RequestException,ValueError) as exc:
            last = exc
            if attempt < 2: time.sleep(attempt+1)
    raise RuntimeError(f'Download or validation failed: {last}')

def preprocess(root):
    root = Path(root)
    source = pd.read_csv(root/'data/raw/POI-MagicK.csv')
    audit = validate(source)
    df = source.set_index('poiID').loc[SELECTED].reset_index().copy()
    # A local tangent-plane projection preserves small-area distances in metres.
    lat0, lon0 = source.lat.mean(), source.long.mean()
    df['x_m'] = np.deg2rad(df.long-lon0)*6371000*np.cos(np.deg2rad(lat0))
    df['y_m'] = np.deg2rad(df.lat-lat0)*6371000
    df['popularity'] = np.log1p(df.n_reviews)/np.log1p(source.n_reviews.max())
    df['base_value'] = .5*(df.rating/5) + .5*df.popularity
    out = root/'data/processed'; out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'attractions.csv',index=False)
    audit.update(selected_rows=len(df),selection_ids=SELECTED,projection_origin=[lat0,lon0],
                 duration_unit_assumption='minutes; raw schema does not declare a unit',
                 unavailable_fields=['opening hours','road network','queue times','visitor preferences'])
    (out/'validation.json').write_text(json.dumps(audit,indent=2))
    return source, df, audit
