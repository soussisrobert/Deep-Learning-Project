#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=HiFT_env
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT

# only the HiFT env is modified (conda install into HiFT only)
conda install -y -n HiFT pillow || exit 1

python -c "import PIL, torch; print(PIL.__version__, torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
