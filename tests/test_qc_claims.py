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
