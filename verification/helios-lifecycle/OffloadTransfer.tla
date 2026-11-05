------------------------ MODULE OffloadTransfer ------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANT RankCount
VARIABLE state

Ranks == 0..(RankCount - 1)
Windows == 1..2
Empty == [accepted |-> {}, done |-> {}, pending |-> FALSE,
          replies |-> 0, error |-> 0, bytes |-> 0]
EmptyState(count) == [windows |-> <<Empty, Empty>>, outstanding |-> [r \in Ranks |-> 0], rankCount |-> count]
Initial == EmptyState(RankCount)
Init == state \in {EmptyState(2), EmptyState(RankCount)}
Active(st) == 0..(st.rankCount - 1)

FirstRefusal(accepted, active) ==
    IF accepted = active THEN 0
    ELSE 1 + CHOOSE r \in active \ accepted : \A s \in active \ accepted : r <= s

Submit(before, after, window, accepted) ==
    /\ accepted \subseteq Active(before)
    /\ before.windows[window] = Empty
    /\ IF accepted = {} THEN after = before
       ELSE after = [before EXCEPT
            !.windows[window].accepted = accepted,
            !.windows[window].pending = TRUE,
            !.windows[window].error = FirstRefusal(accepted, Active(before)),
            !.outstanding = [r \in Ranks |-> @ [r] + IF r \in accepted THEN 1 ELSE 0]]

Complete(before, after, window, rank, failed) ==
    LET job == before.windows[window]
        done == job.done \cup {rank}
    IN /\ rank \in job.accepted \ job.done
       /\ \A earlier \in 1..(window - 1) :
              rank \in before.windows[earlier].accepted => rank \in before.windows[earlier].done
       /\ after = [before EXCEPT
              !.windows[window].done = done,
              !.windows[window].pending = done # job.accepted,
              !.windows[window].replies = IF done = job.accepted THEN 1 ELSE 0,
              !.windows[window].error = IF failed /\ job.error = 0 THEN 4 + rank ELSE job.error,
              !.windows[window].bytes = @ + IF failed THEN 0 ELSE rank + 1,
              !.outstanding[rank] = @ - 1]

Duplicate(before, after, window) ==
    /\ before.windows[window].pending
    /\ after = before

Next == (\E window \in Windows, accepted \in SUBSET Ranks : Submit(state, state', window, accepted))
     \/ (\E window \in Windows, rank \in Ranks, failed \in BOOLEAN : Complete(state, state', window, rank, failed))
     \/ UNCHANGED state

TypeOK == /\ state.rankCount \in {2, RankCount}
          /\ DOMAIN state.outstanding = Ranks
          /\ \A r \in Ranks : state.outstanding[r] \in 0..2
          /\ \A w \in Windows :
               /\ state.windows[w].accepted \subseteq Active(state)
               /\ state.windows[w].done \subseteq state.windows[w].accepted
               /\ state.windows[w].pending \in BOOLEAN
               /\ state.windows[w].replies \in 0..1
               /\ state.windows[w].error \in 0..6
               /\ state.windows[w].bytes \in 0..6
PendingRetainsAcceptedRanks ==
    \A w \in Windows : state.windows[w].pending = (state.windows[w].accepted \ state.windows[w].done # {})
AggregateRequiresEveryAcceptedRank ==
    \A w \in Windows : state.windows[w].replies = 1 =>
        state.windows[w].accepted # {} /\ state.windows[w].done = state.windows[w].accepted
EveryAcceptedJobRepliesOnce ==
    \A w \in Windows : state.windows[w].accepted # {} /\ ~state.windows[w].pending => state.windows[w].replies = 1
PartialAcceptanceReturnsError ==
    \A w \in Windows : state.windows[w].accepted # {} /\ state.windows[w].accepted # Active(state) => state.windows[w].error # 0
OutstandingCountsAcceptedRanks ==
    \A r \in Ranks : state.outstanding[r] = Cardinality({w \in Windows : r \in state.windows[w].accepted \ state.windows[w].done})

Mask(ranks) == (IF 0 \in ranks THEN 1 ELSE 0) + (IF 1 \in ranks THEN 2 ELSE 0) + (IF 2 \in ranks THEN 4 ELSE 0)
Unmask(mask) == {r \in Ranks : (mask \div (2 ^ r)) % 2 = 1}
Fields(job) == <<Mask(job.accepted), Mask(job.done), IF job.pending THEN 1 ELSE 0,
                 job.replies, job.error, job.bytes>>
Projection(st) == Fields(st.windows[1]) \o Fields(st.windows[2]) \o <<st.outstanding[0], st.outstanding[1], st.outstanding[2], st.rankCount>>
Decode(v) ==
    LET job(w) == LET start == (w - 1) * 6
                  IN [accepted |-> Unmask(v[start + 1]), done |-> Unmask(v[start + 2]),
                      pending |-> v[start + 3] = 1, replies |-> v[start + 4],
                      error |-> v[start + 5], bytes |-> v[start + 6]]
    IN [windows |-> <<job(1), job(2)>>, outstanding |-> [r \in Ranks |-> v[13 + r]], rankCount |-> v[16]]

AllowedEdge(edge) ==
    LET before == Decode(edge[1])
        action == edge[2]
        after == Decode(edge[3])
    IN /\ Projection(before) = edge[1]
       /\ Projection(after) = edge[3]
       /\ CASE action[1] = 1 -> Submit(before, after, action[2] + 1, Unmask(action[3]))
            [] action[1] = 2 -> Complete(before, after, action[2] + 1, action[3] % 3, action[3] >= 3)
            [] action[1] = 3 -> before = after
            [] action[1] = 4 -> Duplicate(before, after, action[2] + 1)
            [] OTHER -> FALSE

Spec == Init /\ [][Next]_state
=============================================================================
