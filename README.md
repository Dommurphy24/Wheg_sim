# Wheg Walker: Genetic Algorithm Lab

A graphical 2D simulator for evolving robot gaits. Choose a wheel-leg ("wheg")
or biped from the selector in the control panel (or press **T**). The wheg has
a powered, rotating central hub and three legs arranged around its axis. A
motor drives the hub against the world; gear constraints rigidly couple the
three hip angles. The biped has an upright torso and two independently actuated
legs, each with a hip, knee, and ankle joint. Its torso is slightly enlarged,
both hip joints share the same pivot location, and biped trials end as soon as
the torso capsule contacts the ground.
All legs share the same upper, middle, and foot link lengths. Those lengths
start at 46 units each and evolve together: mutations transfer up to 8 units
between links while preserving the 138-unit total, with each link bounded
between 30 and 65 units. Link mass scales with its length.
The neutral knee and ankle angles are -45 degrees and +60 degrees,
respectively. These are the robot's initial poses, not fixed controller
targets: one evolved knee bias and one ankle bias are shared across the legs,
while each leg retains its own gait amplitude and phase. Initial biases
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
playback speed, **T** switches robot type, and **R** starts a fresh population.
The teal upper leg marks the shared wheg hip drive. Each generation evaluates
every candidate for the same five-second simulated duration on the same
randomly generated terrain profile.
The ground has scrolling distance marks to make forward movement easier to see.
The four fitness-weight sliders are relative weights for maximum forward
speed, forward distance per trial time, reduced torso bounce, and reduced
motor energy use. Releasing a changed slider restarts evolution with a fresh
population so all candidates are scored consistently.

## Architecture

- `whegwalker/physics.py` creates one Pymunk world and advances it with a fixed
  1/120-second control timestep, divided into four fixed physics steps to keep
  fast limb motion stable and reduce floor tunneling. For the wheg, an evolved
  constant-speed motor drives the hub, a shared hip oscillator drives one hip,
  and two gear constraints keep the other hips rigidly synchronized. For the
  biped, both hip oscillators are independently evolved; neither hub rotation
  nor hip coupling is active. Both robots use independent knee and ankle
  oscillators.
- `whegwalker/genetics.py` stores bounded robot-specific gait parameters and
  implements elitism, uniform crossover, and Gaussian mutation.
- `whegwalker/trainer.py` evaluates candidates in a separate worker process,
  so evolution progresses independently of the Pygame event loop.
- `whegwalker/app.py` renders the robot, fitness history, and live training
  status, and handles keyboard controls.

Fitness is the weighted mean of four normalized scores: peak forward speed,
net forward distance per trial time, low torso vertical range during the
trial, and low mean mechanical power used by active motors. The speed and
distance-per-time references are 200 units/s, the bounce reference is 80
units, and the power reference is 1,500,000 simulator units. Default relative
weights are 40, 20, 20, and 20, respectively. A fall subtracts 1.0 from the
normalized fitness score at the instant of a fall; this deduction halves for
every five seconds the robot remained upright. Wheg hub rotation is not treated
as a fall: it is an intended,
powered degree of freedom. Biped genomes evolve independent hip amplitudes and
phases as well as knee and ankle targets; wheg genomes retain their shared
coupled hip drive and hub speed. The controller uses periodic joint targets
rather than a neural network so the relationship between the genome, joint
motion, and locomotion is easy to inspect. The simulation uses simple
randomized terrain and is an educational prototype, not a high-fidelity
robotics model.

The floor is a smoothly interpolated random profile with small height changes.
The robot receives nine terrain-height readings spanning twice its modeled
height ahead in the direction of travel. These readings are displayed in the
viewer and add a bounded knee-lift adjustment when higher ground is detected.
All candidates within a generation use the same terrain profile.
Each physics substep also applies small randomized force and torque
disturbances to the torso. Passing `terrain_seed` and `noise_seed` to
`WhegSimulation` makes terrain and disturbance runs repeatable for testing.

The fall-penalty slider scales the separate fall deduction from 0% to 100% of
the default 1.0-point penalty. At 0%, falls incur no deduction; at 100%, the
full deduction applies to an immediate fall and halves every five seconds
spent upright. The other four sliders continue to weight the normalized
performance metrics.

Joint motors are torque-limited to 750,000 simulator units, but have no maximum
joint-speed cap: their commanded angular rates respond proportionally to target
error. The torque cap was raised from 250,000 after seeded gait trials showed
frequent saturation and lower fitness; a gravity-loaded knee tracking test
confirms the joint motors retain torque headroom. The hub motor is torque-
limited to 2,000,000 to retain reserve when the stronger joints move against
gravity. Every actuated joint also has a damped rotary spring following its
gait target, with stiffness 300, damping 300, and a 50,000 torque cap. This
allows unpowered or torque-overloaded joints to yield during impacts and return
toward their targets. Joint angle limits remain in effect. Torque values are
simulator units, not real-world motor specifications.

## Tests

Run the standard-library tests with:

```powershell
python -m unittest discover -s tests -v
```
