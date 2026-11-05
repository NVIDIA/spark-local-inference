------------------------- MODULE ResponseLifecycle -------------------------
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Capacity, Responses, OutputEvents
VARIABLE state

Cells == 1..Responses
Groups == 1..Responses
EventBudget == OutputEvents + 3
Empty == [group |-> 0, producer |-> 0, reader |-> 0, sent |-> 0,
          received |-> 0, queued |-> 0, terminal |-> 0, cancelled |-> 0]
Owned(row) == row.producer = 1 \/ row.reader = 1
Members(mask) == {r \in Cells : (mask \div (2 ^ (r - 1))) % 2 = 1}
Free(st) == {r \in Cells : ~Owned(st.rows[r])}
GroupOwners(st, g) == {r \in Cells : st.rows[r].group = g /\ Owned(st.rows[r])}

RECURSIVE Sum(_)
Sum(values) == IF Len(values) = 0 THEN 0 ELSE Head(values) + Sum(Tail(values))

Initial == [available |-> Capacity, sizes |-> [g \in Groups |-> 0],
            rows |-> [r \in Cells |-> Empty]]
Init == state = Initial

Admit(before, after, mask, group) ==
    LET members == Members(mask)
        count == Cardinality(members)
    IN /\ mask \in 1..(2 ^ Responses - 1)
       /\ members \subseteq Free(before)
       /\ count <= Capacity
       /\ before.available >= count
       /\ group \in Groups
       /\ before.sizes[group] = 0
       /\ \A g \in Groups : before.sizes[g] = 0 => group <= g
       /\ after = [before EXCEPT !.available = @ - count, !.sizes[group] = count,
                      !.rows = [r \in Cells |->
                          IF r \in members THEN [Empty EXCEPT !.group = group,
                                      !.producer = 1, !.reader = 1]
                          ELSE before.rows[r]]]

Emit(before, after, r) ==
    /\ before.rows[r].producer = 1
    /\ before.rows[r].reader = 1
    /\ before.rows[r].terminal = 0
    /\ before.rows[r].sent < OutputEvents
    /\ after = [before EXCEPT !.rows[r].sent = @ + 1, !.rows[r].queued = @ + 1]

Terminal(before, after, r, terminal) ==
    /\ before.rows[r].producer = 1
    /\ before.rows[r].reader = 1
    /\ before.rows[r].terminal = 0
    /\ after = [before EXCEPT !.rows[r].sent = @ + 1,
                  !.rows[r].queued = @ + 1, !.rows[r].terminal = terminal]

Read(before, after, r) ==
    /\ before.rows[r].reader = 1
    /\ before.rows[r].queued > 0
    /\ after = [before EXCEPT !.rows[r].queued = @ - 1,
                             !.rows[r].received = @ + 1]

Release(before, after, rows, group) ==
    LET occupied == \E r \in Cells : rows[r].group = group /\ Owned(rows[r])
    IN after = [before EXCEPT !.rows = rows,
                    !.sizes[group] = IF occupied THEN @ ELSE 0,
                    !.available = IF occupied THEN @ ELSE @ + before.sizes[group]]

DropReader(before, after, r) ==
    /\ before.rows[r].reader = 1
    /\ LET row == before.rows[r]
           rows == [before.rows EXCEPT ![r] =
               IF row.producer = 0 THEN Empty
               ELSE [row EXCEPT !.reader = 0, !.queued = 0, !.cancelled = 1]]
       IN Release(before, after, rows, row.group)

RetireProducer(before, after, r) ==
    /\ before.rows[r].producer = 1
    /\ before.rows[r].terminal > 0 \/ before.rows[r].cancelled = 1
    /\ LET row == before.rows[r]
           rows == [before.rows EXCEPT ![r] =
               IF row.reader = 0 THEN Empty ELSE [row EXCEPT !.producer = 0]]
       IN Release(before, after, rows, row.group)

Reject(before, after, mask) ==
    /\ mask \in 1..(2 ^ Responses - 1)
    /\ Members(mask) \subseteq Free(before)
    /\ Cardinality(Members(mask)) <= Capacity
    /\ Cardinality(Members(mask)) > before.available
    /\ after = before

ClosedEmit(before, after, r) ==
    /\ before.rows[r].producer = 1
    /\ before.rows[r].reader = 0
    /\ after = before

RepeatFail(before, after, r) ==
    /\ before.rows[r].producer = 1
    /\ before.rows[r].reader = 1
    /\ before.rows[r].terminal = 2
    /\ after = before

Next == (\E mask \in 1..(2 ^ Responses - 1), group \in Groups :
             Admit(state, state', mask, group))
     \/ (\E r \in Cells : Emit(state, state', r)
             \/ Terminal(state, state', r, 1) \/ Terminal(state, state', r, 2)
             \/ Read(state, state', r) \/ DropReader(state, state', r)
             \/ RetireProducer(state, state', r) \/ ClosedEmit(state, state', r)
             \/ RepeatFail(state, state', r))
     \/ (\E mask \in 1..(2 ^ Responses - 1) : Reject(state, state', mask))

WellTyped(st) ==
    /\ st.available \in 0..Capacity
    /\ st.sizes \in [Groups -> 0..Capacity]
    /\ st.rows \in [Cells -> [group : 0..Responses, producer : 0..1, reader : 0..1,
                sent : 0..(OutputEvents + 1), received : 0..(OutputEvents + 1),
                queued : 0..EventBudget, terminal : 0..2, cancelled : 0..1]]

AdmissionAccounted(st) ==
    /\ \A r \in Cells : Owned(st.rows[r]) =>
          /\ st.rows[r].group \in Groups
          /\ st.sizes[st.rows[r].group] >= 1
    /\ \A g \in Groups :
          /\ st.sizes[g] >= Cardinality(GroupOwners(st, g))
          /\ (st.sizes[g] = 0) = (GroupOwners(st, g) = {})
    /\ st.available + Sum([g \in Groups |->
           IF GroupOwners(st, g) = {} THEN 0 ELSE st.sizes[g]]) = Capacity

QueuesBounded(st) ==
    /\ \A r \in Cells :
          /\ st.rows[r].sent <= OutputEvents + IF st.rows[r].terminal = 0 THEN 0 ELSE 1
          /\ st.rows[r].queued <= EventBudget
          /\ st.rows[r].reader = 1 =>
                 st.rows[r].sent = st.rows[r].received + st.rows[r].queued
          /\ st.rows[r].reader = 0 => st.rows[r].queued = 0
    /\ Sum([r \in Cells |-> st.rows[r].queued]) <= Capacity * EventBudget

DroppedReadersCancel(st) ==
    \A r \in Cells : st.rows[r].producer = 1 /\ st.rows[r].reader = 0 =>
         st.rows[r].cancelled = 1

TypeOK == WellTyped(state)
AdmissionRetained == AdmissionAccounted(state)
BufferedOutputBounded == QueuesBounded(state)
DisconnectedProducersCancel == DroppedReadersCancel(state)

RowProjection(row) == <<row.group, row.producer, row.reader, row.sent,
                       row.received, row.queued, row.terminal, row.cancelled>>
RECURSIVE RowsProjection(_, _)
RowsProjection(st, r) == IF r > Responses THEN <<>>
                        ELSE RowProjection(st.rows[r]) \o RowsProjection(st, r + 1)
Projection(st) == <<st.available>> \o [g \in Groups |-> st.sizes[g]]
                  \o RowsProjection(st, 1)

Decode(v) ==
    [available |-> v[1], sizes |-> [g \in Groups |-> v[1 + g]],
     rows |-> [r \in Cells |->
         LET first == Responses + 2 + 8 * (r - 1)
         IN [group |-> v[first], producer |-> v[first + 1], reader |-> v[first + 2],
             sent |-> v[first + 3], received |-> v[first + 4], queued |-> v[first + 5],
             terminal |-> v[first + 6], cancelled |-> v[first + 7]]]]

AllowedEdge(edge) ==
    LET before == Decode(edge[1])
        action == edge[2]
        after == Decode(edge[3])
    IN /\ Projection(before) = edge[1]
       /\ Projection(after) = edge[3]
       /\ WellTyped(before) /\ WellTyped(after)
       /\ AdmissionAccounted(before) /\ AdmissionAccounted(after)
       /\ QueuesBounded(before) /\ QueuesBounded(after)
       /\ DroppedReadersCancel(before) /\ DroppedReadersCancel(after)
       /\ CASE action[1] = 1 -> Admit(before, after, action[2], action[3])
         [] action[1] = 2 -> Emit(before, after, action[2])
         [] action[1] = 3 -> Terminal(before, after, action[2], 1)
         [] action[1] = 4 -> Read(before, after, action[2])
         [] action[1] = 5 -> DropReader(before, after, action[2])
         [] action[1] = 6 -> RetireProducer(before, after, action[2])
         [] action[1] = 7 -> Terminal(before, after, action[2], 2)
         [] action[1] = 8 -> Reject(before, after, action[2])
         [] action[1] = 9 -> ClosedEmit(before, after, action[2])
         [] action[1] = 10 -> RepeatFail(before, after, action[2])
         [] OTHER -> FALSE

Spec == Init /\ [][Next]_state
=============================================================================
