# Low-Rank Matrix Completion over Finite Abelian Group Algebras for Context-Aware Recommendation

The paper is to appear in the Proceeding of ACM International Conference on Information and Knowledge Management (CIKM) 2017.

This repository contains:

1. `Appendix.pdf`: Proof of theorems in the paper.

2. `MCAGA.m`: A MATLAB implementation of MC-AGA, the algorithm proposed in the paper.

3. `demo.m`: A simple demonstration of MC-AGA on a random matrix.

4. Scripts that reproduce our results in the paper.
* For frappe dataset:
    * `MCAGA_frappe.m`: The main script.
    * `metrics.m`: The function for evaluation metrics.

* For sushi dataset:
    * `MCAGA_sushi.m`: The main script.
    * `sushi_explicit2.m`: The script for preprocessing data, including discretization and data splitting.
    * `sushi_explicit3.mat`: The data used by the main script.
    * `rng_state.mat`: The rng state used by the preprocessing script.

The frappe dataset should be downloaded from http://baltrunas.info/data/CARS2_code.zip and placed under the `CARS2_code` directory.

The sushi dataset should be downloaded from http://www.kamishima.net/sushi/sushi3-2016.zip and placed under the `sushi3-2016` directory. If unavailable, try retrieving it from https://archive.org/web/
