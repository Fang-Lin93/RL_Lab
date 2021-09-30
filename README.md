## Implementation of RL Algorithms with Modifications
Have fun!

## Dependencies

- `Python >= 3.8`
- `gym` https://github.com/openai/gym
- `Torch >= 1.9.0`

## Outline (Planned)

```
|-- RL_lab
	|-- README.md
	|-- env		// customized env's 
	|	|-- ??
	    |--
	|-- agents	// agents files
	    |-- human.py // you can play atari game here for fun (WASD... check the keyboard mapping plz)
	    |-- random.py  // random agents
	    |-- dqn.py
	    |-- reinforce.py
	    |-- ...
	    |-- networks  // models
	        |-- naive.py // FC_BN and simple CNN
	        |--...
	|-- checkpoints		//  (local files) save checkpoints on local machines
	|-- test.py  // test the agent by loading from checkpoints
	|-- dqn_trainer.py // temperarily puts trainer here...
	|-- plots.py // 
	|-- ...


```
