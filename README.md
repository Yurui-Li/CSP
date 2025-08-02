Algorithms of MARL 


## Training

To train `MSP` on the SMACv2 and GRF: 

```shell
python3 src/main.py --config=qmix --env-config=sc2_gen_zerg with env_args.capability_config.n_units=5 env_args.capability_config.n_enemies=5 agent=csp mac=csp learner=csp
python3 src/main.py --config=qmix --env-config=gfootball with env_args.map_name=academy_3_vs_1_with_keeper env_args.num_agents=3 agent=csp mac=csp learner=csp
```
