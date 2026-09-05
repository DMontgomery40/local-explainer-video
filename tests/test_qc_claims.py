import math
import pytest
from core.qc_claims import _validate_threshold, _validate_comparative

@pytest.mark.parametrize('value', [9,10,15,20,21])
@pytest.mark.parametrize('claim', ['above','below','outside','within'])
def test_threshold_direction_and_boundaries(value,claim):
    expected={'above':value>20,'below':value<10,'outside':value<10 or value>20,'within':10<=value<=20}[claim]
    assert _validate_threshold({'metric':'x','claim':claim,'range':[10,20]}, {'facts':{'x':value}}).passed is expected

@pytest.mark.parametrize('bounds', [[20,10],[math.nan,20],[10,math.inf],['bad',20],[1]])
def test_invalid_threshold_ranges_fail(bounds):
    assert not _validate_threshold({'metric':'x','claim':'outside','range':bounds},{'facts':{'x':30}}).passed

@pytest.mark.parametrize('baseline,target,actual', [(100,120,20),(100,80,-20),(-100,-80,20),(-100,-120,-20),(100,100,0)])
@pytest.mark.parametrize('direct',[True,False])
@pytest.mark.parametrize('offset',[-2.01,-2,0,2,2.01])
def test_comparative_preserves_sign_and_tolerance(baseline,target,actual,direct,offset):
    pack={'derived':{'x_change':actual}} if direct else {'facts':{'x_session1':baseline,'x_session3':target}}
    for claimed in [actual+offset,-actual+offset]:
        assert _validate_comparative({'metric':'x_change','value':claimed},pack).passed is (abs(claimed-actual)<=2)

def test_comparative_zero_baseline_is_not_a_percentage():
    assert not _validate_comparative({'metric':'x_change','value':0},{'facts':{'x_session1':0,'x_session3':20}}).passed

@pytest.mark.parametrize('kind', ['numeric_vlaue', 'future_type', '', None, 42])
@pytest.mark.parametrize('mixed', [False, True])
def test_unsupported_claims_fail_with_complete_report(tmp_path, kind, mixed):
    import json
    from core.qc_claims import validate_claims, write_claims_report
    claims = [{'scene_id': 7, 'type': kind, 'metric': 'x', 'value': 4}]
    if mixed:
        claims.append({'scene_id': 8, 'type': 'numeric_value', 'metric': 'x', 'value': 4})
    result = validate_claims(claims, {'facts': {'x': 4}})
    assert not result.passed and result.errors
    assert len(result.results) == len(claims)
    assert result.results[0].scene_id == 7 and not result.results[0].passed
    assert 'unsupported' in result.results[0].detail.lower()
    path = tmp_path/'claims.json'; write_claims_report(result, path)
    report = json.loads(path.read_text())
    assert report['total_claims'] == len(claims) and report['failed_claims'] == 1

@pytest.mark.parametrize('predicate', [None, '', 'abvoe', 'not within', 'above below', 'within-ish', 3, ['within']])
@pytest.mark.parametrize('value', [9,15,21])
def test_threshold_rejects_unknown_predicates(predicate,value):
    from core.qc_claims import validate_claims
    result=validate_claims([{'type':'threshold','metric':'x','claim':predicate,'range':[10,20]}],{'facts':{'x':value}})
    assert not result.passed and result.errors

@pytest.mark.parametrize('raw', [{},{'sceens':[]},{'scene_id':1},{'scenes':{}},{'scenes':[{}]},{'scenes':[{'claims':{}}]},{'claims':[4]},[None],None,5,'claims'])
def test_malformed_claim_container_fails(tmp_path,raw):
    import json
    from core.qc_claims import validate_claims_file
    path=tmp_path/'scene_claims.json';path.write_text(json.dumps(raw))
    result=validate_claims_file(path,{'facts':{'x':4}})
    assert not result.passed and result.errors

@pytest.mark.parametrize('shape',['flat','scene','scenes'])
def test_claim_container_valid_shapes_run_claims(tmp_path,shape):
    import json
    from core.qc_claims import validate_claims_file
    claim={'type':'numeric_value','metric':'x','value':4}
    raw=[claim] if shape=='flat' else {'scene_id':7,'claims':[claim]} if shape=='scene' else {'scenes':[{'scene_id':7,'claims':[claim]}]}
    path=tmp_path/'claims.json';path.write_text(json.dumps(raw))
    result=validate_claims_file(path,{'facts':{'x':4}})
    assert result.passed and len(result.results)==1
