from research.adaptive_path.ahe import Event, Outcome, choose_horizon


def history(n=30, family='cash_merger', horizon=180, wins=30):
    events=[]; outcomes=[]
    for k in range(n):
        i=k*400
        eid=f'e{k}'
        events.append(Event(eid,f'T{k}',family,i,100.0))
        outcomes.append(Outcome(eid,horizon,i+horizon,True,k<wins))
    return events,outcomes


def test_no_matured_evidence_means_abstain():
    e=Event('new','ABC','x',1000,10)
    d=choose_horizon(e,[],[])
    assert d.horizon is None


def test_future_outcomes_are_invisible():
    events,outcomes=history()
    e=Event('new','ABC','cash_merger',100,10)
    d=choose_horizon(e,events,outcomes)
    assert d.horizon is None


def test_horizon_never_below_30():
    events,outcomes=history(40,horizon=30,wins=40)
    e=Event('new','ABC','cash_merger',20000,10)
    d=choose_horizon(e,events,outcomes,minimum_blocks=5,target=.80)
    assert d.horizon == 30


def test_missing_matured_outcome_counts_as_failure():
    events,outcomes=history(30,horizon=180,wins=30)
    outcomes[0]=Outcome(outcomes[0].event_id,180,outcomes[0].exit_i,False,None)
    e=Event('new','ABC','cash_merger',20000,10)
    d=choose_horizon(e,events,outcomes,minimum_blocks=5,target=.95)
    assert d.horizon is None


def test_later_horizon_can_be_selected_when_earlier_does_not_clear():
    events=[]; outcomes=[]
    for k in range(40):
        i=k*600; eid=f'e{k}'
        events.append(Event(eid,f'T{k}','family',i,100.0))
        outcomes.append(Outcome(eid,30,i+30,True,k<25))
        outcomes.append(Outcome(eid,180,i+180,True,True))
    e=Event('new','ABC','family',30000,10)
    d=choose_horizon(e,events,outcomes,minimum_blocks=10,target=.80)
    assert d.horizon == 180
