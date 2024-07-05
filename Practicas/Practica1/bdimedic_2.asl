//BOTH TEAMS

//flag_taken sera false tant si es de l'equip axis com si es de l'equip allied i no ha agafat la bandera
+friends_in_fov(ID,TYPE,ANGLE,DIST,HEALTH,POS) : not going_to_heal & not flag_taken
  <-
  +going_to_give_cure;
  !go_to(POS);
  -friends_in_fov(ID,TYPE,ANGLE,DIST,HEALTH,POS).

+packs_in_fov(ID,TYPE,ANGLE,DIST,HEALTH,POS) : health(H) & H < 20
  <-
  .print("health needed");
  +going_to_heal;
  !go_to(POS);
  -packs_in_fov(ID,TYPE,ANGLE,DIST,HEALTH,POS).

+target_reached(T): going_to_give_cure
  <- 
  -going_to_give_cure;
  -target_reached(T);
  -going(T);
  .cure;
  ?going(P);
  !go_to(P).

+target_reached(T): going_to_heal
  <- 
  .print("healed");
  -going_to_heal;
  -target_reached(T);
  -going(T);
  ?going(P);
  !go_to(P).


+!go_to(P)
  <-
  .stop;
  .goto(P);
  +going(P).


//TEAM_AXIS

+flag (F): team(200) 
  <-
  .create_control_points(F,25,4,C);
  +control_points(C);
  .length(C,L);
  +total_control_points(L);
  +patrolling;
  +patroll_point(0);
  .print("Got control points").


+target_reached(T): patrolling & team(200) 
  <-
  -going(T);
  .turn(1.57);
  .turn(1.57);
  .turn(1.57);
  .turn(1.57);
  ?patroll_point(P);
  -+patroll_point(P+1);
  -target_reached(T).

+patroll_point(P): total_control_points(T) & P<T 
  <-
  ?control_points(C);
  .nth(P,C,A);
  !go_to(A).

+patroll_point(P): total_control_points(T) & P==T
  <-
  -patroll_point(P);
  +patroll_point(0).

+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & not going_to_give_cure & not going_to_heal & not chasing
  <-
  .shoot(8, Position);
  .look_at(Position);
  !go_to(Position);
  +chasing;
  +chased(ID);
  -enemies_in_fov(ID,Type,Angle,Distance,Health,Position).


+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & not going_to_give_cure & not going_to_heal & chased(ID)
  <-
  .shoot(8, Position);
  .look_at(Position);
  !go_to(Position);
  -enemies_in_fov(ID,Type,Angle,Distance,Health,Position).


+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(200)
  <-
  .shoot(8, Position);
  -enemies_in_fov(ID,Type,Angle,Distance,Health,Position).


//TEAM_ALLIED

+flag (F): team(100) 
  <-
  .print("going to flag");
  +going_to_flag;
  !go_to(F).

+flag_taken: team(100) 
  <-
  .print("In ASL, TEAM_ALLIED flag_taken");
  ?base(B);
  +returning;
  !go_to(B).

+target_reached(T): team(100) & going_to_flag
  <- 
  .print("arrived to flag");
  -going(T);
  -going_to_flag;
  -target_reached(T).

+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(100)
  <-
  .shoot(8, Position);
  -enemies_in_fov(ID,Type,Angle,Distance,Health,Position).