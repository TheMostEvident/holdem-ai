# Architecture

```text
Environment -> player-specific observation + legal actions
Agent       -> action
Environment -> next state + payoff
Training    -> self-play trajectories + agent update
Evaluation  -> benchmark and tournament results
UI          -> sanitized game view, action input, replay
```

## Constraints

- Maintain strict hidden-information boundaries.
- Test 2–6 player betting rules, heads-up position transitions, and side-pot payouts.
- Carry stacks across hands in tournament mode; eliminate players at zero stack.
- Never assume the initial scaffold is a complete 6-max engine.
