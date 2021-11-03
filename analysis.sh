#!/bin/bash
#SBATCH -J python       # job name
#SBATCH -p defq       # partition name (should always be defq)
#SBATCH -N 1          # number of computing node (should always be 1)
#SBATCH --ntasks=1    # maximum number of parallel tasks (processes) 
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -t 24:00:00     # maximum running time in hh:mm:ss format
#SBATCH -c 1
#SBATCH -w node05

cd "/ImageComMNIST/AnalysisAE/"
python analysis.py --model AE --latent_dim 64 --net_file "results/0224_143540727498_AE.pt"
# python analysis.py --model AE --latent_dim 32 --net_file "results/0224_144714899798_AE.pt"
# python analysis.py --model AE --latent_dim 16 --net_file "results/0224_145903805848_AE.pt"
# python analysis.py --model AE --latent_dim 8 --net_file "results/0224_151045048077_AE.pt"
# python analysis.py --model VAE --latent_dim 64 --net_file "results/0224_144501202108_VAE.pt"
# python analysis.py --model VAE --latent_dim 32 --net_file "results/0224_145647385447_VAE.pt"
# python analysis.py --model VAE --latent_dim 16 --net_file "results/0224_150613640891_VAE.pt"
# python analysis.py --model VAE --latent_dim 8 --net_file "results/0224_152003731589_VAE.pt"
