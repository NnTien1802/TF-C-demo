"""FastAPI functional tests. A random synthetic checkpoint is temporary test-only.
No fake weights are distributed or presented as a trained model.
"""
from io import BytesIO
from pathlib import Path
import sys
import zipfile

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from model_runtime import TFC, Classifier, LABELS

client = TestClient(server.app)


def test_page_and_health():
    assert client.get('/').status_code == 200
    assert '<html lang="vi">' in client.get('/').text
    assert client.get('/static/styles.css').status_code == 200
    assert client.get('/api/health').json()['service'] == 'ok'
    assert len(client.get('/api/labels').json()) == 6


def test_sample_preview_without_model():
    r = client.post('/api/analyze', data={'source': 'example', 'do_predict': 'false'})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload['info']['window_count'] == 12
    assert len(payload['windows'][0]['signal']) == 206
    assert len(payload['windows'][0]['signal'][0]) == 3
    assert payload['windows'][0]['prediction'] is None
    assert payload['windows'][0]['truth']['label'] == 'Gõ phím'
    assert payload['windows'][0]['statistics']['rms'] > 0


def test_csv_preview():
    rng = np.random.default_rng(4)
    vals = rng.normal(size=(410, 3))
    csv = 'x,y,z\n' + '\n'.join(','.join(map(str,row)) for row in vals)
    r = client.post('/api/analyze', data={'source':'upload','is_raw':'true','stride':'100'},
        files={'file':('sensor.csv', csv.encode(), 'text/csv')})
    assert r.status_code == 200, r.text
    obj = r.json()
    assert obj['info']['window_count'] == 3
    assert obj['info']['raw_rows'] == 410
    assert obj['windows'][0]['truth'] is None


def test_zip_preview():
    rows=['seconds_elapsed,x,y,z']
    for i in range(206):rows.append(f'{i/26:.6f},{i/10:.5f},2,3')
    out=BytesIO()
    with zipfile.ZipFile(out,'w') as archive:
        archive.writestr('data/Accelerometer.csv','\n'.join(rows))
    r=client.post('/api/analyze',data={'source':'upload','is_raw':'true'},
      files={'file':('sensor.zip',out.getvalue(),'application/zip')})
    assert r.status_code==200,r.text
    assert len(r.json()['windows'])==1
    assert 25 < r.json()['info']['sampling_hz'] < 27


def test_reject_bad_data():
    r=client.post('/api/analyze', data={'source':'upload'},files={'file':('evil.txt',b'abc','text/plain')})
    assert r.status_code==422
    r=client.post('/api/analyze', data={'source':'upload'},files={'file':('tiny.csv',b'x,y,z\n1,2,3','text/csv')})
    assert r.status_code==422
    r=client.post('/api/analyze', data={'source':'example','stride':'0'})
    assert r.status_code==422


def test_actual_tfc_inference_with_temporary_tensor_checkpoint(tmp_path, monkeypatch):
    import torch
    # Testing class/shape/integration only; randomly initialized model NOT reported as trained.
    temp = tmp_path / 'test_weights.pt'
    encoder, classifier = TFC(), Classifier()
    torch.save({
       'encoder_state_dict': encoder.state_dict(),
       'classifier_state_dict': classifier.state_dict(),
       'config': {'length':206,'channels':3,'labels':LABELS}
    }, temp)
    monkeypatch.setattr(server,'CHECKPOINT',temp)
    server.get_model.cache_clear()
    try:
        r=client.post('/api/analyze', data={'source':'example','do_predict':'true'})
        assert r.status_code==200,r.text
        probs=np.asarray(r.json()['windows'][0]['prediction']['probabilities'])
        assert len(probs)==6
        assert abs(probs.sum()-1) < 1e-4
    finally:
        server.get_model.cache_clear()
