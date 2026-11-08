WAREHOUSE RL PROJECT - SUBMISSION PACKAGE

Files:
1. Warehouse_RL_Four_Deliverables.docx
   - Deliverable 1: one-page MDP problem formulation
   - Deliverable 2: implementation description and run instructions
   - Deliverable 3: actual learning curves and baseline comparison
   - Deliverable 4: five-slide presentation outline

2. warehouse_q_learning.py
   - Complete runnable implementation (NumPy + Matplotlib)

3. Warehouse_RL_5_Slide_Presentation.pptx
   - Exactly five slides: motivation, method, results, limitations, ethics

4. learning_curve.png
5. success_rate_curve.png

Run:
    pip install numpy matplotlib
    python warehouse_q_learning.py

Important:
The reported metrics were generated on the included simplified 5x5 deterministic simulation
with random seed 42. They are not real-world warehouse measurements. The implementation
uses a centralised tabular Q-learning controller that selects one robot's move at each step.
Do not claim DQN, PPO, RLHF or physical robot deployment was implemented.
