------------------------- MODULE CacheOwnership -------------------------
EXTENDS Naturals, Integers, Sequences, FiniteSets

VARIABLE state

Nodes == 1..3
Empty == <<0, 0, 0, 0, 0, 0, 0, 0>>
Initial == [done |-> 0, cursor |-> 0, need |-> <<0, 0, 0>>,
            nodes |-> <<Empty, Empty, Empty>>]
Init == state = Initial

\* Node fields are status, parent, original parent, page mask, readiness fence,
\* last-use fence, direct pins, and retirement fence. Status 0 is unused,
\* 1 pending, 2 ready, 3 retired, and 4 reclaimed. Page IDs are node minus one.
S(st, n) == st.nodes[n][1]
P(st, n) == st.nodes[n][2]
O(st, n) == st.nodes[n][3]
M(st, n) == st.nodes[n][4]
R(st, n) == st.nodes[n][5]
U(st, n) == st.nodes[n][6]
D(st, n) == st.nodes[n][7]
T(st, n) == st.nodes[n][8]
Max(a, b) == IF a > b THEN a ELSE b
Live(st) == {n \in Nodes : S(st, n) \in {1, 2}}
Bits(mask) == {n \in Nodes : (mask \div (2 ^ (n - 1))) % 2 = 1}
Mask(ids) == (IF 1 \in ids THEN 1 ELSE 0)
           + (IF 2 \in ids THEN 2 ELSE 0)
           + (IF 3 \in ids THEN 4 ELSE 0)
Anc(st, n) == IF n = 0 THEN {} ELSE
    {n} \cup (IF P(st, n) = 0 THEN {} ELSE
              {P(st, n)} \cup (IF P(st, P(st, n)) = 0 THEN {} ELSE {P(st, P(st, n))}))
Original(st, n) == {n} \cup (IF O(st, n) = 0 THEN {} ELSE
    {O(st, n)} \cup (IF O(st, O(st, n)) = 0 THEN {} ELSE {O(st, O(st, n))}))
Pages(st, n) == UNION {Bits(M(st, a)) : a \in Anc(st, n)}
Children(st, n) == {c \in Live(st) : P(st, c) = n}
Pinned(st, n) == \E c \in Live(st) : n \in Anc(st, c) /\ D(st, c) > 0
Reads(st, parent, fence) ==
    [st EXCEPT !.nodes = [n \in Nodes |-> IF n \in Anc(st, parent)
        THEN [st.nodes[n] EXCEPT ![6] = Max(@, fence)] ELSE st.nodes[n]],
        !.need = [n \in Nodes |-> IF n \in Pages(st, parent)
                 THEN Max(st.need[n], fence) ELSE st.need[n]]]

Publish(before, after, n, encoded) ==
    LET parent == IF n = 1 THEN 0 ELSE IF n = 2 THEN 1 ELSE (encoded \div 3) + 1
        fence == encoded % 3
        read == Reads(before, parent, fence)
    IN /\ n = before.cursor + 1 /\ n \in Nodes
       /\ parent < n /\ (IF parent = 0 THEN TRUE ELSE S(before, parent) = 2)
       /\ after = [read EXCEPT !.cursor = n, !.need[n] = fence,
          !.nodes[n] = <<IF fence <= before.done THEN 2 ELSE 1,
                        parent, parent, 2 ^ (n - 1), fence, fence, 0, 0>>]

Pin(before, after, n) ==
    /\ S(before, n) = 2 /\ D(before, n) = 0
    /\ after = [before EXCEPT !.nodes[n][7] = 1]
Unpin(before, after, n) ==
    /\ n \in Live(before) /\ D(before, n) = 1
    /\ after = [before EXCEPT !.nodes[n][7] = 0]
Use(before, after, n, fence) ==
    /\ S(before, n) = 2 /\ fence \in 0..2
    /\ after = Reads(before, n, fence)
Duplicate(before, after, n, fence) ==
    /\ n \in Live(before) /\ fence \in 0..2
    /\ after = Reads(before, P(before, n), fence)

Retire(before, after, n) ==
    after = [before EXCEPT !.nodes[n][1] = 3, !.nodes[n][2] = 0,
             !.nodes[n][7] = 0, !.nodes[n][8] = Max(U(before, n), R(before, n))]
Cancel(before, after, n) ==
    /\ S(before, n) = 1 /\ ~Pinned(before, n) /\ Children(before, n) = {}
    /\ Retire(before, after, n)
Evict(before, after, n) ==
    /\ S(before, n) = 2
    /\ IF Children(before, n) = {} THEN
          /\ ~Pinned(before, n)
          /\ Retire(before, after, n)
       ELSE /\ D(before, n) = 0 /\ Cardinality(Children(before, n)) = 1
            /\ LET child == CHOOSE c \in Children(before, n) : TRUE
                   retired == [before EXCEPT !.nodes[n][1] = 3, !.nodes[n][2] = 0,
                       !.nodes[n][4] = 0, !.nodes[n][8] = Max(U(before, n), R(before, n))]
               IN after = [retired EXCEPT !.nodes[child][2] = P(before, n),
                   !.nodes[child][4] = Mask(Bits(M(before, n)) \cup Bits(M(before, child))),
                   !.nodes[child][6] = Max(U(before, child), Max(U(before, n), R(before, n)))]
Complete(before, after, fence) ==
    /\ fence \in before.done..2
    /\ after = [before EXCEPT !.done = fence,
        !.nodes = [n \in Nodes |-> IF S(before, n) = 1 /\ R(before, n) <= fence
                    THEN [before.nodes[n] EXCEPT ![1] = 2]
                    ELSE IF S(before, n) = 3 /\ T(before, n) <= fence
                    THEN [before.nodes[n] EXCEPT ![1] = 4] ELSE before.nodes[n]]]

Step(before, after) ==
    (\E n \in Nodes, encoded \in 0..5 : Publish(before, after, n, encoded))
    \/ (\E n \in Nodes : Pin(before, after, n) \/ Unpin(before, after, n)
                         \/ Cancel(before, after, n) \/ Evict(before, after, n))
    \/ (\E n \in Nodes, fence \in 0..2 : Use(before, after, n, fence)
                                            \/ Duplicate(before, after, n, fence))
    \/ (\E fence \in 0..2 : Complete(before, after, fence))
Next == Step(state, state')

TypeOK == /\ state.done \in 0..2 /\ state.cursor \in 0..3
          /\ \A n \in Nodes : /\ S(state, n) \in 0..4 /\ D(state, n) \in 0..1
             /\ P(state, n) \in 0..(n - 1) /\ O(state, n) \in 0..(n - 1)
             /\ M(state, n) \in 0..7 /\ R(state, n) \in 0..2 /\ U(state, n) \in 0..2
             /\ T(state, n) \in 0..2 /\ state.need[n] \in 0..2
ImmutableDescendantPages == \A n \in Live(state) : Pages(state, n) = Original(state, n)
DisjointPageOwnership == \A a, b \in Nodes :
    (a # b /\ S(state, a) \in {1, 2, 3} /\ S(state, b) \in {1, 2, 3})
    => Bits(M(state, a)) \cap Bits(M(state, b)) = {}
InheritedLastUse == \A n \in Live(state) : \A page \in Bits(M(state, n)) : U(state, n) >= state.need[page]
RetirementRetainsReads == \A n \in Nodes : S(state, n) \in {3, 4} =>
    /\ T(state, n) >= U(state, n)
    /\ \A page \in Bits(M(state, n)) : T(state, n) >= state.need[page]
ReclaimAfterCompletion == \A n \in Nodes : S(state, n) = 4 => T(state, n) <= state.done
PendingParentProtection == \A n \in Live(state) : P(state, n) = 0 \/ P(state, n) \in Live(state)

Projection(st) == <<st.done, st.cursor>> \o st.need \o st.nodes[1] \o st.nodes[2] \o st.nodes[3]
Decode(v) == [done |-> v[1], cursor |-> v[2], need |-> SubSeq(v, 3, 5),
              nodes |-> <<SubSeq(v, 6, 13), SubSeq(v, 14, 21), SubSeq(v, 22, 29)>>]
AllowedEdge(edge) ==
    LET before == Decode(edge[1])
        action == edge[2]
        after == Decode(edge[3])
    IN /\ Projection(before) = edge[1] /\ Projection(after) = edge[3]
       /\ CASE action[1] = 1 -> Publish(before, after, action[2], action[3])
            [] action[1] = 2 -> Pin(before, after, action[2])
            [] action[1] = 3 -> Unpin(before, after, action[2])
            [] action[1] = 4 -> Use(before, after, action[2], action[3])
            [] action[1] = 5 -> Cancel(before, after, action[2])
            [] action[1] = 6 -> Evict(before, after, action[2])
            [] action[1] = 7 -> Complete(before, after, action[3])
            [] action[1] = 8 -> Duplicate(before, after, action[2], action[3])
            [] action[1] = 9 -> after = before
            [] OTHER -> FALSE
Spec == Init /\ [][Next]_state
=============================================================================
