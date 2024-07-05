//GENERAL



+friends_in_fov(ID,Type,Angle,Distance,Health,Position): not assisting & Health < 80 & not flag_taken
  <- 
  .print("FIF 1");
  !go_to(Position);
  +assisting.

+health(X): X < 75
  <-
  .cure;
  .wait(250).

+packs_in_fov(ID, 1001, Ang, Dist, Health, Pos): health(X) & X < 50
  <-
  .stop;
  !go_to(Pos).

+packs_in_fov(ID, 1002, Ang, Dist, Health, Pos): ammo(X) & X < 25 
  <-
  .stop;
  !go_to(Pos).


+!go_to(P) <- .stop; .goto(P); +going(P).


+target_reached(T): assisting
  <-
  .print("MEDPACK here!");
  .cure;
  -assisting;
  -going(T);
  -target_reached(T);
  ?going(P);
  !go_to(P).



+target_reached(T): chasing
  <-
  -chasing;
  -chased(_);
  -target_reached(T);
  ?going(P);
  !go_to(P).



//TEAM_AXIS

+flag (F): team(200)
  <-
  .create_control_points(F,35,4,C);
  +control_points(C);
  .length(C,L);
  +total_control_points(L);
  +patrolling;
  +patroll_point(0);
  .print("Got control points").



+target_reached(T): patrolling & team(200) 
  <-
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

+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & not assisting & not chasing
  <-
  ?threshold_shots(X);
  .shoot(X, Position);
  .stop;
  .goto(Position);
  .look_at(Position);
  +chasing;
  +chased(ID).


+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & not assisting & chased(ID)
  <-
  ?threshold_shots(X);
  .shoot(X, Position);
  .stop;
  .goto(Position);
  .look_at(Position).


+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(200)
  <-
  ?threshold_shots(X);
  .shoot(X, Position).


//TEAM_ALLIED 

+flag (F): team(100)
  <-
  +not_following;
  .turn(0.5);
  .turn(-1);
  !go_to(F).
  

+flag_taken: team(100) 
  <-
  .print("In ASL, TEAM_ALLIED flag_taken");
  ?base(B);
  .goto(B).


+target_reached(T): flag(T) & team(100)
  <-
  -going(T);
  -target_reached(T);
  ?base(B);
  !go_to(B).

  
+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(100)
  <-
  ?threshold_shots(X);
  .shoot(X,Position).


+friends_in_fov(ID,Type,Angle,Distance,Health,Position): team(100) & not assisting & not chasing
  <-
  .stop;
  .goto(Position);
  .look_at(Position);
  +chasing;
  +chased(ID).

+friends_in_fov(ID,Type,Angle,Distance,Health,Position): team(100) & not assisting & chased(ID)
  <-
  .stop;
  .goto(Position);
  .look_at(Position).