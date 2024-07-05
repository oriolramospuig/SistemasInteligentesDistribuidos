//TEAM_AXIS

+flag (F): team(200)
  <-
  .create_control_points(F,25,3,C);
  +control_points(C);
  .length(C,L);
  +total_control_points(L);             
  +patrolling;                          
  +patroll_point(0);                    
  .print("Got control points").

+target_reached(T): patrolling & team(200)
  <-
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
  .goto(A).                             

+patroll_point(P): total_control_points(T) & P==T
  <-
  -patroll_point(P);                   
  +patroll_point(0).   

+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & not healing & not reloading & not chasing
  <-
  ?threshold_shots(X);
  .shoot(X,Position);
  .look_at(Position);
  .goto(Position);
  +chasing;
  +chased(ID);
  -enemies_in_fov(ID,Type,Angle,Distance,Health,Position).


+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & not healing & not reloading & chased(ID)
  <-
  ?threshold_shots(X);
  .shoot(X,Position);
  .look_at(Position);
  .goto(Position);
  -enemies_in_fov(ID,Type,Angle,Distance,Health,Position).


+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(200)
  <-
  ?threshold_shots(X);
  .shoot(X,Position);
  -enemies_in_fov(ID,Type,Angle,Distance,Health,Position). 



//TEAM_ALLIED

+flag (F): team(100)
  <-
  .print("going to flag");
  +goingToFlag;
  .goto(F).                           

+flag_taken: team(100)
  <-
  .print("In ASL, TEAM_ALLIED flag_taken");
  ?base(B);                             
  +returning;                      
  .goto(B);                            
  -exploring.                          

+heading(H): exploring
  <-
  .wait(2000);                        
  .turn(0.375).

+target_reached(T): team(100) & goingToFlag
  <-
  .print("flag reached");
  -goingToFlag;
  +exploring;                        
  .turn(0.375).

+enemies_in_fov(ID,Type,Angle,Distance,Health,Position): team(100)
  <-
  ?threshold_shots(X);
  .shoot(X,Position);
  -enemies_in_fov(ID,Type,Angle,Distance,Health,Position).



// PER QUALSEVOL SOLDAT   

//empieza plan de cura
+packs_in_fov(ID, 1001, Angle, Distance, Health, PosMedPack) : health(H) & (H < 30) & (not reloading)
  <- 
  .print("salud critica");
  .stop;
  !urgentCure.

+!urgentCure : packs_in_fov(ID, 1001, Angle, Distance, Health, PosMedPack)
  <- 
  +healing;
  .goto(PosMedPack).

+!urgentCure
  <- 
  .turn(1.57).

+target_reached(T) : healing
  <-
  .print("soldado curado");
  -healing;
  -target_reached(T);
  !!go_to_flag.
//acaba plan de cura

//empieza plan de recarga
+packs_in_fov(ID, 1002, Angle, Distance, Health, PosAmmo) : ammo(X) & (X < 30) & (not healing)
  <-
  .print("municion baja");
  .stop;
  !urgentReload.

+!urgentReload : packs_in_fov(ID, 1002, Angle, Distance, Health, PosAmmo)
  <- 
  +reloading;
  .goto(PosAmmo).

+!urgentReload
  <- 
  .turn(1.57).

+target_reached(T) : reloading
  <-
  .print("arma recargada");
  -reloading;
  -target_reached(T);
  !!go_to_flag.
//acaba plan de recarga

+!go_to_flag
  <-
  ?flag(F);
  .goto(F).