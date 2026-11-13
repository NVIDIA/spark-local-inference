-------------------------- MODULE Speculation --------------------------
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxSubmissions, PageTokens, ContextTokens, OutputTokens,
          Batch, Drafts, Width, Packed, DraftPages, PromptTokens
VARIABLE state

Min(a, b) == IF a < b THEN a ELSE b
Max(a, b) == IF a > b THEN a ELSE b
CeilPages(tokens, page) == (tokens + page - 1) \div page
Zero == <<0, 0>>
Geometry(st) == st[1]
Lanes(st) == 1..Geometry(st)[4]
QueueLength(st) == IF st[4] = Zero THEN 0 ELSE IF st[5] = Zero THEN 1 ELSE 2
Advance(st, ceiling) == Min(ceiling, Geometry(st)[7])
Pending(st, slot) == Advance(st, st[4][slot]) + Advance(st, st[5][slot])
Limit(st, slot) ==
    Min(Min(Geometry(st)[6], Geometry(st)[5] + 1),
        Min(Geometry(st)[2] - st[slot + 1][3],
            Geometry(st)[3] - st[slot + 1][2] - Pending(st, slot)))

Initial ==
    << <<PageTokens, ContextTokens, OutputTokens, Batch, Drafts, Width, Packed, DraftPages>>,
       <<PromptTokens, 1, PromptTokens, CeilPages(PromptTokens, PageTokens) + DraftPages>>,
       IF Batch = 2 THEN <<PromptTokens, 1, PromptTokens,
                           CeilPages(PromptTokens, PageTokens) + DraftPages>> ELSE <<0, 0, 0, 0>>,
       Zero, Zero, <<0, 0>> >>
Init == state = Initial

Commit(before, after, ceilings) ==
    /\ before[6][1] < MaxSubmissions
    /\ QueueLength(before) < 2
    /\ \A s \in Lanes(before) : ceilings[s] = Limit(before, s) /\ ceilings[s] > 0
    /\ \A s \in (1..2) \ Lanes(before) : ceilings[s] = 0
    /\ after =
        <<before[1],
          <<before[2][1], before[2][2], before[2][3] + Advance(before, ceilings[1]),
             Max(before[2][4], CeilPages(before[2][3] + ceilings[1], Geometry(before)[1]) + Geometry(before)[8])>>,
          IF 2 \in Lanes(before) THEN
            <<before[3][1], before[3][2], before[3][3] + Advance(before, ceilings[2]),
              Max(before[3][4], CeilPages(before[3][3] + ceilings[2], Geometry(before)[1]) + Geometry(before)[8])>>
          ELSE before[3],
          IF QueueLength(before) = 0 THEN ceilings ELSE before[4],
          IF QueueLength(before) = 0 THEN Zero ELSE ceilings,
          <<before[6][1] + 1, before[6][2]>> >>

Complete(before, after, emitted) ==
    /\ QueueLength(before) > 0
    /\ \A s \in Lanes(before) : emitted[s] \in 1..Advance(before, before[4][s])
    /\ \A s \in (1..2) \ Lanes(before) : emitted[s] = 0
    /\ emitted[1] + emitted[2] <= Geometry(before)[7]
    /\ after =
        <<before[1],
          <<before[2][1] + emitted[1], before[2][2] + emitted[1],
            before[2][1] + emitted[1] + Advance(before, before[5][1]), before[2][4]>>,
          IF 2 \in Lanes(before) THEN
            <<before[3][1] + emitted[2], before[3][2] + emitted[2],
              before[3][1] + emitted[2] + Advance(before, before[5][2]), before[3][4]>>
          ELSE before[3],
          before[5], Zero, <<before[6][1], before[6][2] + 1>> >>

Reject(before, after) == after = before
Next == (\E a, b \in 0..Width : Commit(state, state', <<a, b>>))
     \/ (\E a, b \in 0..Width : Complete(state, state', <<a, b>>))
     \/ Reject(state, state')

TypeOK ==
    /\ state[6][1] \in 0..MaxSubmissions
    /\ state[6][2] \in 0..state[6][1]
    /\ state[6][1] - state[6][2] = QueueLength(state)
    /\ \A s \in Lanes(state) :
        /\ state[s + 1][1] \in PromptTokens..Geometry(state)[2]
        /\ state[s + 1][2] \in 1..Geometry(state)[3]
        /\ state[s + 1][3] \in state[s + 1][1]..Geometry(state)[2]
        /\ state[s + 1][4] \in 1..(CeilPages(Geometry(state)[2], Geometry(state)[1]) + Geometry(state)[8])

PendingUsesPhysicalCapacity ==
    \A s \in Lanes(state) : state[s + 1][3] = state[s + 1][1]
        + Min(state[4][s], Geometry(state)[7]) + Min(state[5][s], Geometry(state)[7])

OutputAndContextBounded ==
    \A s \in Lanes(state) :
        /\ state[s + 1][2] + Pending(state, s) <= Geometry(state)[3]
        /\ state[s + 1][3] <= Geometry(state)[2]

FullLaneStorageRetained ==
    \A s \in Lanes(state) :
        LET last == IF state[5] # Zero THEN state[5][s] ELSE state[4][s]
            through == state[s + 1][3] + last - Advance(state, last)
        IN state[s + 1][4] >= CeilPages(through, Geometry(state)[1]) + Geometry(state)[8]

AllowedEdge(edge) ==
    LET before == edge[1]
        action == edge[2]
        after == edge[3]
    IN CASE action[1] = 1 -> Commit(before, after, <<action[2], action[3]>>)
         [] action[1] = 2 -> Complete(before, after, <<action[2], action[3]>>)
         [] action[1] = 3 -> Reject(before, after)
         [] OTHER -> FALSE

Spec == Init /\ [][Next]_state
=============================================================================
