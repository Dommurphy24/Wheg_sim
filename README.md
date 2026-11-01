# Wheg Walker: Genetic Algorithm Lab

A graphical 2D simulator for evolving the gait of a wheel-leg ("wheg") robot.
The robot has a powered, rotating central hub and three two-link legs arranged
around its axis. A motor drives the hub against the world; each hip pivots at
the hub, and gear constraints rigidly couple the three hip angles. Each leg
also has an independently actuated knee and ankle joint.
All three legs share the same upper, middle, and foot link lengths. Those
lengths start at 46 units each and evolve together: mutations transfer up to
8 units between links while preserving the 138-unit total, with each link
bounded between 30 and 65 units. Link mass scales with its length.
The neutral knee and ankle angles are -45 degrees and +60 degrees,
respectively. These are the robot's initial poses, not fixed controller
targets: one evolved knee bias and one ankle bias are shared across all three
legs, while each leg retains its own gait amplitude and phase. Initial biases
vary broadly around the neutral angles, mutation can explore the full joint
travel, and each angle mutation applies only a small bias back toward neutral.

## Run

Use Python 3.10 or newer:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

Training starts automatically in a separate worker process, independent of
the Pygame display loop. The viewer continuously animates the best gait found
so far and refreshes the generation/evaluation/fitness status as candidates
finish; candidate trials are simulated without drawing. **Space** pauses or
resumes evolution, **B** restarts the visible best gait, **1/2/3** change its
playback speed, and **R** starts a fresh population. The teal upper leg marks
the shared hip drive. Each generation evaluates every candidate for the same
five-second simulated duration.
The ground has scrolling distance marks to make forward movement easier to see.

## Architecture

- `whegwalker/physics.py` creates one Pymunk world and advances it with a fixed
  1/120-second control timestep, divided into four fixed physics steps to keep
  fast limb motion stable and reduce floor tunneling. An evolved constant-speed
  motor drives the hub. A shared hip oscillator drives one hip; two gear
  constraints keep the other hips rigidly synchronized. Independent oscillators
  drive each knee and ankle, giving each leg three articulated degrees of
  freedom.
- `whegwalker/genetics.py` stores bounded hub and gait parameters and
  implements elitism, uniform crossover, and Gaussian mutation.
- `whegwalker/trainer.py` evaluates candidates in a separate worker process,
  so evolution progresses independently of the Pygame event loop.
- `whegwalker/app.py` renders the robot, fitness history, and live training
  status, and handles keyboard controls.

Fitness rewards forward displacement and penalizes the hub dropping below the
floor threshold. Hub rotation is not treated as a fall: it is an intended,
powered degree of freedom. The controller evolves periodic hip/knee targets
and hub speed rather than a neural network so the relationship between the
genome, joint motion, and locomotion is easy to inspect. The simulation uses a
simple flat floor and is an educational prototype, not a high-fidelity
robotics model.

Joint motors are capped at 750,000 simulator torque units, with an 8 rad/s
maximum commanded rate. The cap was raised from 250,000 after seeded gait
trials showed frequent saturation and lower fitness; a gravity-loaded knee
tracking test confirms the joint motors retain torque headroom. The hub motor
is capped at 2,000,000 to retain torque reserve when the stronger joints are
moving against gravity. Torque values are simulator units, not real-world
motor specifications.

## Tests

Run the standard-library tests with:

```powershell
python -m unittest discover -s tests -v
```
