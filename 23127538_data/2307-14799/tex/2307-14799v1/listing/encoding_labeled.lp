#const sub_size = 0.  % if "sub_size" is positive, it limits machines in subgroups #(\label{prg:encoding:1}#)
#const lot_step = 0.  % if "lot_step" is set to 1, operations of lot get separated #(\label{prg:encoding:2}#)
#const by_setup = 0.  % if "by_setup" is set to 1, operation setups get subdivided #(\label{prg:encoding:3}#)

% split tool groups into subgroups of at most "sub_size" many machines in each

subgroup_size(sub_size) :- sub_size > 0. #(\label{prg:encoding:7}#)

subgroup(G, 1, M)      :- tool(G, M), sub_size = 0. #(\label{prg:encoding:9}#)
subgroup(G, J, M)      :- tool(G, N), subgroup_size(S), not tool(G, N+1), #(\label{prg:encoding:10}#)
                          J = 1 .. K, K = (N+S-1)/S, L = (N\K+J-|N\K-J|)/2,
                          M = (J-1)*(N/K)+(L+J-1-|L-J+1|)/2+1 .. J*(N/K)+L. #(\label{prg:encoding:12}#)
subgroup(G, J, M1, M2) :- subgroup(G, J, M1), not subgroup(G, J, M1-1), #(\label{prg:encoding:13}#)
                          subgroup(G, J, M2), not subgroup(G, J, M2+1). #(\label{prg:encoding:14}#)

% map each production operation to some subgroup of the respective tool group

step_index(G, K, (L, P, I, T, S)) :- lot(L, P), route(P, I, G, T, M, N, S), #(\label{prg:encoding:18}#)
                             K = #count{P', L', I' * lot_step : lot(L', P'),
                                           route(P', I', G, T', M', N', S'),
                             (P', L', I' * lot_step) < (P, L, I * lot_step)}. #(\label{prg:encoding:21}#)

step_subgroup(G, K\J+1, O) :- step_index(G, K, O), #(\label{prg:encoding:23}#)
                              subgroup(G, J, _), not subgroup(G, J+1, _). #(\label{prg:encoding:24}#)

assignable(O, G, M) :- step_subgroup(G, J, O), subgroup(G, J, M), by_setup = 0. #(\label{prg:encoding:26}#)

% distinguish machines in each subgroup based on production operations' setups

subgroup_setup(G, J, S)    :- step_subgroup(G, J, (L, P, I, T, S)), by_setup != 0. #(\label{prg:encoding:30}#)
subgroup_setup(G, J, S, N) :- subgroup_setup(G, J, S), #(\label{prg:encoding:31}#)
                     N = #sum+{T, L, P, I : step_subgroup(G, J, (L, P, I, T, S))}. #(\label{prg:encoding:32}#)

setup_index(G, J, K, S, N) :- subgroup_setup(G, J, S, N), #(\label{prg:encoding:34}#)
          K = #count{S', N' : subgroup_setup(G, J, S', N'), (N, S) < (N', S')}. #(\label{prg:encoding:35}#)
setup_index(G, J, K, N)    :- setup_index(G, J, K, S, N). #(\label{prg:encoding:36}#)
setup_index(G, J, K)       :- setup_index(G, J, K-1, _), #(\label{prg:encoding:37}#)
                              not setup_index(G, J, K, _). #(\label{prg:encoding:38}#)

setup_machine(G, J, M, M\K, N) :- subgroup(G, J, M1, M2), setup_index(G, J, K), #(\label{prg:encoding:40}#)
                                  setup_index(G, J, M\K, N), M = 0 .. M2-M1. #(\label{prg:encoding:41}#)
setup_machine(G, J, M, K, A+N) :- subgroup(G, J, M1, M2), setup_index(G, J, K, N), #(\label{prg:encoding:42}#)
                                  next_machine(G, J, M2-M1, K-1, A, M). #(\label{prg:encoding:43}#)

%* [...] *% #(\label{prg:encoding:45}#)

next_machine(G, J, K, K, A, K)  :- subgroup(G, J, M1, M2), K = M2-M1,
                                   setup_machine(G, J, K, K, A),
                                   setup_index(G, J, K+1, N).
next_machine(G, J, 0, K, A, 0)  :- load_machine(G, J, 0, K, A).
next_machine(G, J, M, K, A, M1) :- load_machine(G, J, M, K, A1),
                                   next_machine(G, J, M-1, K, A2, M2),
                                   A = (A1+A2-|A1-A2|)/2,
                                   M1 = #min{M : A = A1; M2 : A = A2}.

load_machine(G, J, M, K+1, A) :- subgroup(G, J, M1, M2), setup_index(G, J, K+2, N),
                                 setup_machine(G, J, M, M, A), K = M2-M1, M < K.
load_machine(G, J, M, K, A+N) :- subgroup(G, J, M1, M2), setup_index(G, J, K, N),
                                 next_machine(G, J, M2-M1, K-1, A, M),
                                 not setup_index(G, J, K+1).
load_machine(G, J, M, K, A)   :- subgroup(G, J, M1, M2),
                                 next_machine(G, J, M2-M1, K-1, A', M'),
                                 load_machine(G, J, M, K-1, A), M != M',
                                 not setup_index(G, J, K+1).

assignable((L, P, I, T, S), G, M1+M) :- step_subgroup(G, J, (L, P, I, T, S)), #(\label{prg:encoding:47}#)
                                        setup_machine(G, J, M, K, A),
                                        setup_index(G, J, K, S, N),
                                        subgroup(G, J, M1, M2). #(\label{prg:encoding:50}#)

% generate batches for operations of different lots with same product in subgroups

step_batch(G, J, M, N, (L, P, I, T, S)) :- #(\label{prg:encoding:54}#)
      step_subgroup(G, J, (L, P, I, T, S)), route(P, I, G, T, M, N, S), 1 < N. #(\label{prg:encoding:55}#)
lots_batch((L1, P, I, T, S), (L2, P, I, T, S)) :- #(\label{prg:encoding:56}#)
      step_batch(G, J, M, N, (L1, P, I, T, S)),
      step_batch(G, J, M, N, (L2, P, I, T, S)), L1 < L2. #(\label{prg:encoding:58}#)

{batch(G, O1, O2) : lots_batch(O1, O2)} 1 :- step_batch(G, J, M, N, O2). #(\label{prg:encoding:60}#)
:- step_batch(G, J, M, N, O1), N #count{O2 : batch(G, O1, O2)}. #(\label{prg:encoding:61}#)

batched(O1, O2; O2, O1; O2) :- batch(G, O1, O2). #(\label{prg:encoding:63}#)
:- batched(O1), batch(G, O1, O2). #(\label{prg:encoding:64}#)

% generate assignment of production operations to machines and order of processing

{assign(O, G, M) : assignable(O, G, M)} = 1 :- step_index(G, K, O). #(\label{prg:encoding:68}#)

step_pair(G, O1, O2)   :- assignable(O1, G, M), assignable(O2, G, M), O1 < O2. #(\label{prg:encoding:70}#)
step_assign(G, O1, O2) :- step_pair(G, O1, O2), assign(O1, G, M), assign(O2, G, M). #(\label{prg:encoding:71}#)
lots_assign(G, O1, O2) :- O1 = (L1, P1, I1, T1, S1), step_assign(G, O1, O2), #(\label{prg:encoding:72}#)
                          O2 = (L2, P2, I2, T2, S2), (L1, P1) < (L2, P2). #(\label{prg:encoding:73}#)
in_sequence(G, O1, O2) :- lots_assign(G, O1, O2), not batch(G, O1, O2). #(\label{prg:encoding:74}#)
:- batch(G, O1, O2), not lots_assign(G, O1, O2). #(\label{prg:encoding:75}#)

{lots_order(G, O1, O2)} :- in_sequence(G, O1, O2). #(\label{prg:encoding:77}#)
 lots_order(G, O2, O1)  :- in_sequence(G, O1, O2), not lots_order(G, O1, O2). #(\label{prg:encoding:78}#)

% determine setup and maintenance operations required before production operations

main_setup(G, H, K, W) :- pm(G, H, E, M, N, W), #(\label{prg:encoding:82}#)
          K = #count{H' : pm(G, H', E', M', N', W'), (W, H) <= (W', H')}. #(\label{prg:encoding:83}#)
main_setup(G, S, 0, U) :- setup(G, S, U, M). #(\label{prg:encoding:84}#)

step_order(G, O1, O2) :- main_setup(G, X, K, V), lots_order(G, O1, O2). #(\label{prg:encoding:86}#)
step_order(G, O1, O2) :- main_setup(G, X, K, V), step_assign(G, O1, O2), #(\label{prg:encoding:87}#)
                         O1 = (L, P, I1, T1, S1), O2 = (L, P, I2, T2, S2). #(\label{prg:encoding:88}#)

%* [...] *% #(\label{prg:encoding:90}#)

change(G, O, O1) :- setup(G, S, U, M), step_order(G, O, O1), O = (L, P, I, T, S),
                    O1 = (L1, P1, I1, T1, S1), S1 != 0, S1 != S.
ignore(G, O, O2) :- change(G, O, O1), step_order(G, O1, O2), O = (L, P, I, T, S),
                    O2 = (L2, P2, I2, T2, S2), 1 #count{ : S2 = 0; : S2 = S},
                    I < I2 : (L, P) = (L2, P2).
reuse(G, O, O2)  :- setup(G, S, U, M), step_order(G, O, O2), O = (L, P, I, T, S),
                    O2 = (L2, P2, I2, T2, S2), 1 #count{ : S2 = 0; : S2 = S},
                    not ignore(G, O, O2).
reuse(G, O2)     :- reuse(G, O, O2), O2 = (L2, P2, I2, T2, S2), S2 != 0.

equip(O, G, S) :- setup(G, S, U, M), step_index(G, K, O), O = (L, P, I, T, S), #(\label{prg:encoding:92}#)
                  not reuse(G, O). #(\label{prg:encoding:93}#)

{maintain(O, G, H)} :- pm(G, H, E, M, N, W), step_index(G, K, O). #(\label{prg:encoding:95}#)
:- batched(O1, O2), maintain(O1, G, H), not maintain(O2, G, H). #(\label{prg:encoding:96}#)
:- pm(G, H, E, M, N, W), step_index(G, K, O), not step_order(G, _, O), #(\label{prg:encoding:97}#)
   not maintain(O, G, H). #(\label{prg:encoding:98}#)

%* [...] *% #(\label{prg:encoding:100}#)

repeat(G, H, O, O2) :- pm(G, H, E, M, N, W), step_order(G, O, O2),
                       maintain(O2, G, H).
repeat(G, H, O, O2) :- repeat(G, H, O, O1), step_order(G, O1, O2),
                       O = (L, P, I, T, S),
                       O2 = (L2, P2, I2, T2, S2), I < I2 : (L, P) = (L2, P2).

repeat(G, H, O)     :- repeat(G, H, O, O2). #(\label{prg:encoding:102}#)
bundle(G, H, O, O2) :- pm(G, H, E, M, N, W), step_order(G, O, O2), #(\label{prg:encoding:103}#)
                       not repeat(G, H, O, O2). #(\label{prg:encoding:104}#)
:- pm(G, H, lots, M, N, W), step_index(G, K, O), #(\label{prg:encoding:105}#)
   N #count{O2 : bundle(G, H, O, O2); O2 : batched(O, O2)}. #(\label{prg:encoding:106}#)
:- pm(G, H, lots, M, N, W), repeat(G, H, O), maintain(O, G, H), #(\label{prg:encoding:107}#)
   #count{O2 : bundle(G, H, O, O2); O2 : batched(O, O2)} < M-1. #(\label{prg:encoding:108}#)
:- pm(G, H, time, M, N, W), step_index(G, K, O), #(\label{prg:encoding:109}#)
   O = (L, P, I, T, S), #sum+{T2, O2 : bundle(G, H, O, O2),
                                 O2 = (L2, P2, I2, T2, S2), not batched(O2)} > N-T. #(\label{prg:encoding:111}#)
:- pm(G, H, time, M, N, W), repeat(G, H, O), maintain(O, G, H), #(\label{prg:encoding:112}#)
   O = (L, P, I, T, S), #sum+{T2, O2 : bundle(G, H, O, O2),
                                 O2 = (L2, P2, I2, T2, S2), not batched(O2)} < M-T. #(\label{prg:encoding:114}#)

delay(O, G, 0, U)   :- main_setup(G, S, 0, U), equip(O, G, S). #(\label{prg:encoding:116}#)
delay(O, G, 0, U+V) :- main_setup(G, S, 0, U), equip(O, G, S), delay(O, G, 1, V). #(\label{prg:encoding:117}#)
delay(O, G, K, W)   :- main_setup(G, H, K, W), maintain(O, G, H), 0 < K. #(\label{prg:encoding:118}#)
delay(O, G, K, W+V) :- main_setup(G, H, K, W), maintain(O, G, H), 0 < K, #(\label{prg:encoding:119}#)
                       delay(O, G, K+1, V). #(\label{prg:encoding:120}#)
delay(O, G, K-1, V) :- delay(O, G, K, V), 0 < K.

% assert difference logic constraints on completion times of production operations

&diff{0 - (L, P, 1, T, S)} <= -T   :- step_index(G, K, (L, P, 1, T, S)), #(\label{prg:encoding:125}#)
                                      not setup(G, S, _, _). #(\label{prg:encoding:126}#)
&diff{0 - (L, P, I, T, S)} <= -T-U :- step_index(G, K, (L, P, I, T, S)), #(\label{prg:encoding:127}#)
                                      setup(G, S, U, M). #(\label{prg:encoding:128}#)

&diff{O1 - O2} <= 0    :- batched(O1,O2). #(\label{prg:encoding:130}#)
&diff{O1 - O2} <= -T   :- step_index(G1, K1, O1), O1 = (L, P, I, T1, S1), #(\label{prg:encoding:131}#)
                          step_index(G2, K2, O2), O2 = (L, P, I+1, T, S). #(\label{prg:encoding:132}#)
&diff{O1 - O2} <= -T   :- lots_order(G, O1, O2), O2 = (L, P, I, T, S). #(\label{prg:encoding:133}#)
&diff{O1 - O2} <= -T-V :- step_order(G, O1, O2), O2 = (L, P, I, T, S), #(\label{prg:encoding:134}#)
                          delay(O2, G, 0, V). #(\label{prg:encoding:135}#)

&diff{O - makespan} <= 0 :- step_index(G, K, O), #(\label{prg:encoding:137}#)
                            O = (L, P, I, T, S), not route(P, I+1, _, _, _, _, _). #(\label{prg:encoding:138}#)

% subprograms for minimizing the makespan, setup and batch violations of schedules

#program opt(b). #(\label{prg:encoding:142}#)
#external bound(b). #(\label{prg:encoding:143}#)

&diff{makespan - 0} <= b :- bound(b). #(\label{prg:encoding:145}#)

#program weak(b). #(\label{prg:encoding:147}#)
bound(b). #(\label{prg:encoding:148}#)

change(G, O) :- change(G, O, O1). #(\label{prg:encoding:150}#)

:~ setup(G, S, U, M), equip(O, G, S), change(G, O), #(\label{prg:encoding:152}#)
   #count{O2 : reuse(G, O, O2), not batched(O2)} < M-1. [1@2, O] #(\label{prg:encoding:153}#)

:~ step_batch(G, J, M, N, O), not batched(O), #(\label{prg:encoding:155}#)
   #count{O2 : batch(G, O, O2)} M-2. [1@1, O] #(\label{prg:encoding:156}#)

% output production operations' machines and the setup and maintenance operations

#show assign/3.
#show equip/3.
#show maintain/3.
