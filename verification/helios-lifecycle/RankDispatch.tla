-------------------------- MODULE RankDispatch --------------------------
EXTENDS Naturals, FiniteSets

CONSTANT PeerCount
VARIABLE state

Peers == 1..PeerCount
Ranks == 0..PeerCount
Initial == [sent |-> 0, localStarted |-> FALSE, done |-> {},
            failed |-> FALSE, published |-> FALSE, released |-> FALSE]
Init == state = Initial

SendPeer(before, after) ==
    /\ ~before.failed
    /\ before.sent < PeerCount
    /\ after = [before EXCEPT !.sent = @ + 1]

SubmitLocal(before, after) ==
    /\ ~before.failed
    /\ before.sent = PeerCount
    /\ ~before.localStarted
    /\ after = [before EXCEPT !.localStarted = TRUE]

DispatchFailure(before, after) ==
    /\ ~before.failed
    /\ ~before.published
    /\ after = [before EXCEPT !.failed = TRUE]

FinishRank(before, after, rank) ==
    /\ rank \notin before.done
    /\ IF rank = 0 THEN before.localStarted ELSE rank <= before.sent
    /\ after = [before EXCEPT !.done = @ \cup {rank}]

Publish(before, after) ==
    /\ ~before.failed
    /\ ~before.published
    /\ before.localStarted
    /\ before.done = Ranks
    /\ after = [before EXCEPT !.published = TRUE]

Release(before, after) ==
    /\ ~before.failed
    /\ before.published
    /\ after = [before EXCEPT !.released = TRUE]

Next == SendPeer(state, state')
     \/ SubmitLocal(state, state')
     \/ DispatchFailure(state, state')
     \/ (\E rank \in Ranks : FinishRank(state, state', rank))
     \/ Publish(state, state')
     \/ Release(state, state')
     \/ UNCHANGED state

TypeOK == /\ state.sent \in 0..PeerCount
          /\ state.localStarted \in BOOLEAN
          /\ state.done \subseteq Ranks
          /\ state.failed \in BOOLEAN
          /\ state.published \in BOOLEAN
          /\ state.released \in BOOLEAN
LocalFollowsEverySend == state.localStarted => state.sent = PeerCount
FailureRetainsOwnership == state.failed => ~state.released
PublicationRequiresEveryRank == state.published => state.done = Ranks /\ ~state.failed
OnlyAcceptedRanksFinish == \A rank \in state.done : IF rank = 0 THEN state.localStarted ELSE rank <= state.sent

Spec == Init /\ [][Next]_state
=============================================================================
