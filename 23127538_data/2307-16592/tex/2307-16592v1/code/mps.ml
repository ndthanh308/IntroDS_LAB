open Bsml
open Stdlib.Base

let list_of_par (v: 'a par) : 'a list = 
  List.map (proj v) procs

let map : ('a->'b) -> 'a list par -> 'b list par = 
  fun f v -> parfun (List.map f) v
    
let reduce : ('a->'a->'a) -> 'a -> 'a list par -> 'a = 
  fun op e v ->
  let locally_reduced = parfun (List.fold_left op e) v in 
  List.fold_left op e (list_of_par locally_reduced)

let mps : int list par -> int = 
  let op  (xm, xs) (ym, ys) = (max 0 (max xm (xs+ym)), xs+ys) in  
  let f x = (max 0 x, x) in
  fun v -> fst (reduce op (0, 0) (map f v))

let value = 
  mps (mkpar(function 0->[1;2]|1->[-1;2]|2->[-1;3]|3 -> [-4]| _ -> []))
let test = (value = 6)


(* requires : input list non empty *)
let maximum = function 
  | [] -> assert false
  | h::t -> List.fold_left max h t 
  
let sum = List.fold_left (+) 0

let rec prefix = 
  function
    | [] -> [[]]
    | x::xs -> []::(List.map (fun l->x::l) (prefix xs))

(* for the requires of maximim to hold, prove that: 
   - forall l, prefix l is always non empty 
   - if l is non empty then forall f, map f l is non empty *)
let mps_spec l = 
  maximum(List.map sum (prefix l))
  
let to_list: 'a list par -> 'a list = 
  fun v -> 
  List.flatten(list_of_par v)
  
(* Correctness to express as an ensures
   A first step could be to prove that 
   mps v = mps_seq (to_list v) 
   where mps_seq is the same as mps but 
   reduce replaced by List.fold_left and map by List.map.
   Then prove that mps_seq returns the same thing
   than mps_spec. *)  
let always_true (v: int list par) = 
    mps v = mps_spec (to_list v)
    
let init : int -> (int->'a) -> 'a list = 
  fun size f -> 
  Array.to_list(Array.init size f)

let generate () = 
  let f _ = (Random.int 100) - 50 in
  Random.self_init();
  mkpar(fun pid->init ((Random.int 30)+5) f)