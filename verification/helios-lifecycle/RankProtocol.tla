------------------------- MODULE RankProtocol -------------------------
EXTENDS Naturals, Sequences, FiniteSets

CONSTANT MaxSubmissions
VARIABLE state

Peers == {0, 1}
Ranks == {0, 1, 2}
Initial == [submitted |-> 0, removed |-> 0, removedDone |-> {}, queue |-> <<>>]
Init == state = Initial

Reserve(before, after, graph, barrier) ==
    /\ before.submitted < MaxSubmissions
    /\ Len(before.queue) < 2
    /\ IF Len(before.queue) = 0 THEN TRUE ELSE before.queue[Len(before.queue)].committed
    /\ barrier = 0 \/ Len(before.queue) = 0
    /\ \A i \in 1..Len(before.queue) : before.queue[i].barrier = 0
    /\ LET fence == before.submitted + 1
           frame == [fence |-> fence, graph |-> graph, barrier |-> barrier,
                     ready |-> {}, committed |-> FALSE, done |-> {}]
       IN after = [before EXCEPT !.submitted = fence, !.queue = Append(@, frame)]

Ready(before, after, fence, peer) ==
    \E i \in 1..Len(before.queue) :
        /\ before.queue[i].fence = fence
        /\ ~before.queue[i].committed
        /\ peer \notin before.queue[i].ready
        /\ after = [before EXCEPT !.queue[i].ready = @ \cup {peer}]

Commit(before, after, fence) ==
    \E i \in 1..Len(before.queue) :
        /\ before.queue[i].fence = fence
        /\ ~before.queue[i].committed
        /\ before.queue[i].ready = Peers
        /\ after = [before EXCEPT !.queue[i].committed = TRUE]

Complete(before, after, fence, rank) ==
    \E i \in 1..Len(before.queue) :
        /\ before.queue[i].fence = fence
        /\ rank \notin before.queue[i].done
        /\ IF rank = 0 THEN TRUE
           ELSE /\ before.queue[i].committed
                /\ \A j \in 1..(i - 1) : rank \in before.queue[j].done
        /\ after = [before EXCEPT !.queue[i].done = @ \cup {rank}]

Remove(before, after) ==
    /\ Len(before.queue) > 0
    /\ Head(before.queue).done = Ranks
    /\ after = [before EXCEPT !.removed = Head(before.queue).fence,
                             !.removedDone = Head(before.queue).done,
                             !.queue = Tail(@)]

Next == (\E graph \in 0..1, barrier \in 0..1 : Reserve(state, state', graph, barrier))
     \/ (\E fence \in 1..MaxSubmissions, peer \in Peers : Ready(state, state', fence, peer))
     \/ (\E fence \in 1..MaxSubmissions : Commit(state, state', fence))
     \/ (\E fence \in 1..MaxSubmissions, rank \in Ranks : Complete(state, state', fence, rank))
     \/ Remove(state, state')
     \/ UNCHANGED state

TypeOK == /\ state.submitted \in 0..MaxSubmissions
          /\ state.removed \in 0..state.submitted
          /\ state.removedDone \subseteq Ranks
          /\ Len(state.queue) <= 2
          /\ \A i \in 1..Len(state.queue) :
              /\ state.queue[i].fence \in 1..state.submitted
              /\ state.queue[i].graph \in 0..1
              /\ state.queue[i].barrier \in 0..1
              /\ state.queue[i].ready \subseteq Peers
              /\ state.queue[i].committed \in BOOLEAN
              /\ state.queue[i].done \subseteq Ranks

CommitRequiresEveryPeer ==
    \A i \in 1..Len(state.queue) : state.queue[i].committed => state.queue[i].ready = Peers
RemovalRequiresEveryRank == state.removed > 0 => state.removedDone = Ranks
OrderedRemoval ==
    /\ \A i \in 1..Len(state.queue) : state.queue[i].fence > state.removed
    /\ \A i \in 1..(Len(state.queue) - 1) : state.queue[i].fence < state.queue[i + 1].fence
FeedbackIsExclusive ==
    \A i \in 1..Len(state.queue) : state.queue[i].barrier = 1 => Len(state.queue) = 1

PeerMask(peers) == (IF 0 \in peers THEN 1 ELSE 0) + (IF 1 \in peers THEN 2 ELSE 0)
RankMask(ranks) == PeerMask(ranks) + (IF 2 \in ranks THEN 4 ELSE 0)
Unmask(mask, ids) == {i \in ids : (mask \div (2 ^ i)) % 2 = 1}
Fields(frame) == <<frame.fence, frame.graph, frame.barrier, PeerMask(frame.ready),
                   IF frame.committed THEN 1 ELSE 0, RankMask(frame.done)>>
FrameFields(st, i) == IF i <= Len(st.queue) THEN Fields(st.queue[i]) ELSE <<0, 0, 0, 0, 0, 0>>
Projection(st) == <<st.submitted, st.removed, Len(st.queue)>>
                  \o FrameFields(st, 1) \o FrameFields(st, 2)

Decode(v) ==
    LET frame(i) == LET start == 3 + (i - 1) * 6
                    IN [fence |-> v[start + 1], graph |-> v[start + 2],
                        barrier |-> v[start + 3], ready |-> Unmask(v[start + 4], Peers),
                        committed |-> v[start + 5] = 1, done |-> Unmask(v[start + 6], Ranks)]
    IN [submitted |-> v[1], removed |-> v[2], removedDone |-> IF v[2] = 0 THEN {} ELSE Ranks,
        queue |-> IF v[3] = 0 THEN <<>> ELSE IF v[3] = 1 THEN <<frame(1)>> ELSE <<frame(1), frame(2)>>]

AllowedEdge(edge) ==
    LET before == Decode(edge[1])
        action == edge[2]
        after == Decode(edge[3])
    IN /\ Projection(before) = edge[1]
       /\ Projection(after) = edge[3]
       /\ CASE action[1] = 1 -> Reserve(before, after, action[2], action[3])
            [] action[1] = 2 -> Ready(before, after, action[2], action[3])
            [] action[1] = 3 -> Commit(before, after, action[2])
            [] action[1] = 4 -> Complete(before, after, action[2], action[3])
            [] action[1] = 5 -> Remove(before, after)
            [] action[1] = 6 -> after = before
            [] OTHER -> FALSE

Spec == Init /\ [][Next]_state
=============================================================================
