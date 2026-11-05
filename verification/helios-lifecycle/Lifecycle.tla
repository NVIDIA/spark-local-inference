---------------------------- MODULE Lifecycle ----------------------------
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxSubmissions, MaxGeneration
VARIABLE state

Slots == {0, 1}
Free == 0
Live == 1
Retiring == 2

Members(mask) == CASE mask = 1 -> {0}
                  [] mask = 2 -> {1}
                  [] mask = 3 -> Slots
Mask(members) == (IF 0 \in members THEN 1 ELSE 0)
                + (IF 1 \in members THEN 2 ELSE 0)

Initial == [submitted |-> 0, completed |-> 0,
            phase |-> [s \in Slots |-> Live],
            generation |-> [s \in Slots |-> 1],
            lastUse |-> [s \in Slots |-> 0], queue |-> <<>>]
Init == state = Initial

Admit(before, after, slot) ==
    /\ before.phase[slot] = Free
    /\ before.generation[slot] < MaxGeneration
    /\ after = [before EXCEPT !.phase[slot] = Live,
                             !.generation[slot] = @ + 1]

Commit(before, after, mask, barrier) ==
    LET selected == Members(mask)
        fence == before.submitted + 1
        pending == [fence |-> fence, members |-> selected,
                    generation |-> [s \in selected |-> before.generation[s]],
                    barrier |-> barrier]
    IN /\ before.submitted < MaxSubmissions
       /\ Len(before.queue) < 2
       /\ \A s \in selected : before.phase[s] = Live
       /\ \A i \in 1..Len(before.queue) : before.queue[i].barrier = 0
       /\ barrier = 0 \/ Len(before.queue) = 0
       /\ after = [before EXCEPT !.submitted = fence,
                      !.lastUse = [s \in Slots |->
                          IF s \in selected THEN fence ELSE before.lastUse[s]],
                      !.queue = Append(@, pending)]

Complete(before, after) ==
    /\ Len(before.queue) > 0
    /\ LET fence == Head(before.queue).fence
           released == {s \in Slots : before.phase[s] = Retiring
                                     /\ before.lastUse[s] <= fence}
       IN after = [before EXCEPT !.completed = fence,
                         !.queue = Tail(@),
                         !.phase = [s \in Slots |->
                             IF s \in released THEN Free ELSE before.phase[s]],
                         !.lastUse = [s \in Slots |->
                             IF s \in released THEN 0 ELSE before.lastUse[s]]]

Cancel(before, after, slot) ==
    /\ before.phase[slot] = Live
    /\ LET waiting == before.lastUse[slot] > before.completed
       IN after = [before EXCEPT !.phase[slot] = IF waiting THEN Retiring ELSE Free,
                                !.lastUse[slot] = IF waiting THEN @ ELSE 0]

UnconfirmedDrain(before, after) == after = before
ConfirmedDrain(before, after) ==
    after = [before EXCEPT !.completed = before.submitted, !.queue = <<>>,
                          !.phase = [s \in Slots |-> Free],
                          !.lastUse = [s \in Slots |-> 0]]

Next == (\E s \in Slots : Admit(state, state', s))
     \/ (\E mask \in 1..3, barrier \in 0..1 : Commit(state, state', mask, barrier))
     \/ Complete(state, state')
     \/ (\E s \in Slots : Cancel(state, state', s))
     \/ UnconfirmedDrain(state, state')
     \/ ConfirmedDrain(state, state')

TypeOK == /\ state.submitted \in 0..MaxSubmissions
          /\ state.completed \in 0..state.submitted
          /\ state.phase \in [Slots -> {Free, Live, Retiring}]
          /\ state.generation \in [Slots -> 1..MaxGeneration]
          /\ state.lastUse \in [Slots -> 0..state.submitted]
          /\ Len(state.queue) <= 2

OrderedCompletions ==
    /\ \A i \in 1..Len(state.queue) :
        state.completed < state.queue[i].fence
        /\ state.queue[i].fence <= state.submitted
    /\ \A i \in 1..(Len(state.queue) - 1) :
        state.queue[i].fence < state.queue[i + 1].fence

QueuedReferencesRetainOwnership ==
    \A i \in 1..Len(state.queue) :
        \A s \in state.queue[i].members :
            /\ state.phase[s] # Free
            /\ state.generation[s] = state.queue[i].generation[s]
            /\ state.lastUse[s] >= state.queue[i].fence

RetirementWaitsForCompletion ==
    \A s \in Slots : state.phase[s] = Retiring => state.lastUse[s] > state.completed

FeedbackIsExclusive ==
    \A i \in 1..Len(state.queue) : state.queue[i].barrier = 1 => Len(state.queue) = 1

QueueFence(st, i) == IF i <= Len(st.queue) THEN st.queue[i].fence ELSE 0
QueueMask(st, i) == IF i <= Len(st.queue) THEN Mask(st.queue[i].members) ELSE 0
QueueBarrier(st, i) == IF i <= Len(st.queue) THEN st.queue[i].barrier ELSE 0

Projection(st) == <<st.submitted, st.completed, st.phase[0], st.phase[1],
                    st.generation[0], st.generation[1], st.lastUse[0], st.lastUse[1],
                    QueueFence(st, 1), QueueFence(st, 2),
                    QueueMask(st, 1), QueueMask(st, 2),
                    QueueBarrier(st, 1), QueueBarrier(st, 2)>>

Decode(v) ==
    LET generation == [s \in Slots |-> v[5 + s]]
        entry(i) == [fence |-> v[8 + i], members |-> Members(v[10 + i]),
                     generation |-> [s \in Members(v[10 + i]) |-> generation[s]],
                     barrier |-> v[12 + i]]
    IN [submitted |-> v[1], completed |-> v[2],
        phase |-> [s \in Slots |-> v[3 + s]], generation |-> generation,
        lastUse |-> [s \in Slots |-> v[7 + s]],
        queue |-> IF v[9] = 0 THEN <<>>
                  ELSE IF v[10] = 0 THEN <<entry(1)>> ELSE <<entry(1), entry(2)>>]

AllowedEdge(edge) ==
    LET before == Decode(edge[1])
        action == edge[2]
        after == Decode(edge[3])
    IN /\ Projection(before) = edge[1]
       /\ Projection(after) = edge[3]
       /\ CASE action[1] = 1 -> Admit(before, after, action[2])
         [] action[1] = 2 -> Commit(before, after, action[2], action[3])
         [] action[1] = 3 -> Complete(before, after)
         [] action[1] = 4 -> Cancel(before, after, action[2])
         [] action[1] = 5 -> UnconfirmedDrain(before, after)
         [] action[1] = 6 -> ConfirmedDrain(before, after)
         [] OTHER -> FALSE

Spec == Init /\ [][Next]_state
=============================================================================
