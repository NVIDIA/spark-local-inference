-------------------- MODULE CheckpointReservation --------------------
EXTENDS Naturals, FiniteSets, Sequences

CONSTANT RequestCount, RowCount, MaxFence
VARIABLE state

Requests == 1..RequestCount
Rows == 1..RowCount
Free == 0
Guard == 1
Armed == 2
Copying == 3
Published == 4
Retired == 5
Mandatory == 6
Empty == [phase |-> Free, owner |-> 0, fence |-> 0]
Initial == [rows |-> [row \in Rows |-> Empty],
            live |-> Requests, submitted |-> 0, completed |-> 0,
            uses |-> [row \in Rows |-> 0]]
Init == state = Initial

Owned(st, request) == {row \in Rows : st.rows[row].owner = request
                                  /\ st.rows[row].phase \in {Guard, Armed, Copying}}
Captures(st, request) == {row \in Owned(st, request) : st.rows[row].phase \in {Armed, Copying}}
Guards(st, request) == {row \in Owned(st, request) : st.rows[row].phase = Guard}

ReserveGuard(before, after, request, row) ==
    /\ request \in before.live
    /\ Guards(before, request) = {}
    /\ before.rows[row].phase = Free
    /\ after = [before EXCEPT !.rows[row] = [phase |-> Guard, owner |-> request, fence |-> 0]]

ArmGuard(before, after, row) ==
    /\ before.rows[row].phase = Guard
    /\ Captures(before, before.rows[row].owner) = {}
    /\ after = [before EXCEPT !.rows[row].phase = Armed]

ReserveExtra(before, after, request, row) ==
    /\ request \in before.live
    /\ Captures(before, request) = {}
    /\ before.rows[row].phase = Free
    /\ after = [before EXCEPT !.rows[row] = [phase |-> Armed, owner |-> request, fence |-> 0]]

Produce(before, after, row) ==
    /\ before.rows[row].phase = Armed
    /\ before.rows[row].fence = 0
    /\ before.submitted < MaxFence
    /\ after = [before EXCEPT !.submitted = @ + 1,
                              !.rows[row].fence = before.submitted + 1,
                              !.uses[row] = before.submitted + 1]

StartCopy(before, after, row) ==
    /\ before.rows[row].phase = Armed
    /\ before.rows[row].fence > 0
    /\ before.rows[row].fence <= before.completed
    /\ before.submitted < MaxFence
    /\ after = [before EXCEPT !.submitted = @ + 1,
                              !.rows[row].phase = Copying,
                              !.rows[row].fence = before.submitted + 1,
                              !.uses[row] = before.submitted + 1]

Complete(before, after) ==
    /\ before.completed < before.submitted
    /\ after = [before EXCEPT !.completed = @ + 1]

Publish(before, after, row) ==
    /\ before.rows[row].phase = Copying
    /\ before.rows[row].fence <= before.completed
    /\ after = [before EXCEPT !.rows[row].phase = Published]

Cancel(before, after, request) ==
    /\ request \in before.live
    /\ after = [before EXCEPT !.live = @ \ {request},
        !.rows = [row \in Rows |->
            IF row \in Owned(before, request)
            THEN IF before.rows[row].phase = Guard THEN Empty
                 ELSE [before.rows[row] EXCEPT !.phase = Retired]
            ELSE before.rows[row]]]

ReleaseGuard(before, after, row) ==
    /\ before.rows[row].phase = Guard
    /\ after = [before EXCEPT !.rows[row] = Empty]

MandatoryAllocation(before, after, row) ==
    /\ before.rows[row].phase \in {Free, Guard}
    /\ after = [before EXCEPT !.rows[row] = [phase |-> Mandatory, owner |-> 0, fence |-> 0]]

ReleaseMandatory(before, after, row) ==
    /\ before.rows[row].phase = Mandatory
    /\ after = [before EXCEPT !.rows[row] = Empty]

Evict(before, after, row) ==
    /\ before.rows[row].phase = Published
    /\ after = [before EXCEPT !.rows[row].phase = Retired]

Reclaim(before, after, row) ==
    /\ before.rows[row].phase = Retired
    /\ before.rows[row].fence <= before.completed
    /\ after = [before EXCEPT !.rows[row] = Empty]

OptionalGuard(before, after, request) ==
    before = after \/ (\E row \in Rows : ReserveGuard(before, after, request, row))

ArmAndProduce(before, after, request, row) ==
    \E guarded \in {before} \cup
        {[before EXCEPT !.rows[guard] = [phase |-> Guard, owner |-> request, fence |-> 0]] : guard \in Rows} :
        /\ OptionalGuard(before, guarded, request)
        /\ LET armed == [guarded EXCEPT !.rows[row] = [phase |-> Armed, owner |-> request, fence |-> 0]]
           IN /\ (ReserveExtra(guarded, armed, request, row) \/ ArmGuard(guarded, armed, row))
              /\ Produce(armed, after, row)

CompleteAndPublish(before, after, row) ==
    LET completed == [before EXCEPT !.completed = @ + 1]
    IN Complete(before, completed) /\ Publish(completed, after, row)

CancelAndReclaim(before, after, request) ==
    LET cancelled == [before EXCEPT !.live = @ \ {request},
        !.rows = [row \in Rows |->
            IF row \in Owned(before, request)
            THEN IF before.rows[row].phase = Guard THEN Empty
                 ELSE [before.rows[row] EXCEPT !.phase = Retired]
            ELSE before.rows[row]]]
    IN /\ Cancel(before, cancelled, request)
       /\ after = [cancelled EXCEPT !.rows = [row \in Rows |->
            IF cancelled.rows[row].phase = Retired /\ cancelled.rows[row].fence <= cancelled.completed
            THEN Empty ELSE cancelled.rows[row]]]

CompleteAndReclaim(before, after) ==
    LET completed == [before EXCEPT !.completed = @ + 1]
    IN /\ Complete(before, completed)
       /\ after = [completed EXCEPT !.rows = [row \in Rows |->
            IF completed.rows[row].phase = Retired /\ completed.rows[row].fence <= completed.completed
            THEN Empty ELSE completed.rows[row]]]

Next == (\E request \in Requests, row \in Rows : ReserveGuard(state, state', request, row))
     \/ (\E row \in Rows : ArmGuard(state, state', row))
     \/ (\E request \in Requests, row \in Rows : ReserveExtra(state, state', request, row))
     \/ (\E row \in Rows : Produce(state, state', row))
     \/ (\E row \in Rows : StartCopy(state, state', row))
     \/ Complete(state, state')
     \/ (\E row \in Rows : Publish(state, state', row))
     \/ (\E request \in Requests : Cancel(state, state', request))
     \/ (\E row \in Rows : MandatoryAllocation(state, state', row))
     \/ (\E row \in Rows : ReleaseMandatory(state, state', row))
     \/ (\E row \in Rows : Evict(state, state', row))
     \/ (\E row \in Rows : Reclaim(state, state', row))
     \/ (\E request \in Requests, row \in Rows : ArmAndProduce(state, state', request, row))
     \/ (\E row \in Rows : CompleteAndPublish(state, state', row))
     \/ (\E request \in Requests : CancelAndReclaim(state, state', request))
     \/ CompleteAndReclaim(state, state')
     \/ (\E row \in Rows : ReleaseGuard(state, state', row))
     \/ UNCHANGED state

TypeOK == /\ state.live \subseteq Requests
          /\ state.completed \in 0..state.submitted
          /\ state.submitted \in 0..MaxFence
          /\ \A row \in Rows :
              /\ state.rows[row].phase \in Free..Mandatory
              /\ state.rows[row].owner \in 0..RequestCount
              /\ state.rows[row].fence \in 0..state.submitted
              /\ state.uses[row] \in 0..state.submitted
OutstandingUseRetainsRow == \A row \in Rows :
    state.uses[row] > state.completed => state.rows[row].phase \in {Armed, Copying, Retired}
PublicationWaitsForCopy == \A row \in Rows :
    state.rows[row].phase = Published => state.rows[row].fence <= state.completed
OneGuardAndCapturePerRequest == \A request \in Requests :
    Cardinality(Guards(state, request)) <= 1 /\ Cardinality(Captures(state, request)) <= 1
CancelledRequestsHaveNoReservation == \A request \in Requests \ state.live : Owned(state, request) = {}
GuardsHaveNoGPUUse == \A row \in Rows : state.rows[row].phase = Guard => state.rows[row].fence = 0

FromProjection(p) ==
    [submitted |-> p[1], completed |-> p[2],
     live |-> {request \in Requests : (p[3] \div (2 ^ (request - 1))) % 2 = 1},
     rows |-> [row \in Rows |-> [phase |-> p[4 + (row - 1) * 4],
                                   owner |-> p[5 + (row - 1) * 4],
                                   fence |-> p[6 + (row - 1) * 4]]],
     uses |-> [row \in Rows |-> p[7 + (row - 1) * 4]]]

AllowedEdge(edge) ==
    LET before == FromProjection(edge[1])
        action == edge[2]
        after == FromProjection(edge[3])
    IN CASE action[1] = 1 -> ReserveGuard(before, after, action[2], action[3])
         [] action[1] = 2 -> ArmGuard(before, after, action[3])
         [] action[1] = 3 -> Produce(before, after, action[3])
         [] action[1] = 4 -> StartCopy(before, after, action[3])
         [] action[1] = 5 -> Complete(before, after)
         [] action[1] = 6 -> Publish(before, after, action[3])
         [] action[1] = 7 -> Cancel(before, after, action[2])
         [] action[1] = 8 -> MandatoryAllocation(before, after, action[3])
         [] action[1] = 9 -> ReleaseMandatory(before, after, action[3])
         [] action[1] = 10 -> Evict(before, after, action[3])
         [] action[1] = 11 -> Reclaim(before, after, action[3])
         [] action[1] = 12 -> ReserveExtra(before, after, action[2], action[3])
         [] action[1] = 13 -> ArmAndProduce(before, after, action[2], action[3])
         [] action[1] = 14 -> CompleteAndPublish(before, after, action[3])
         [] action[1] = 15 -> CancelAndReclaim(before, after, action[2])
         [] action[1] = 16 -> CompleteAndReclaim(before, after)
         [] action[1] = 17 -> ReleaseGuard(before, after, action[3])
         [] OTHER -> FALSE

Spec == Init /\ [][Next]_state
=============================================================================
