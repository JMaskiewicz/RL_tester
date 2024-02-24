import subprocess
import os
import torch

def get_repository_root_path():
    try:
        # Run the git command to get the top-level directory
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT).strip().decode('utf-8')
        return repo_root
    except subprocess.CalledProcessError as e:
        print("Error getting repository root:", e.output.decode())
        return None


def save_model(self, base_dir="saved models", sub_dir="DDQN", file_name="ddqn"):
    # Get the repository root path
    repo_root = get_repository_root_path()
    if repo_root is None:
        print("Repository root not found. Using current directory as the base path.")
        repo_root = "."

    # Construct the full path
    full_path = os.path.join(repo_root, base_dir, sub_dir, file_name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Define the full file paths for the policy and target models
    policy_path = f"{full_path}_policy.pt"
    target_path = f"{full_path}_target.pt"

    # Save the models
    torch.save(self.q_policy.state_dict(), policy_path)
    torch.save(self.q_target.state_dict(), target_path)
    print(f"Models saved successfully: {policy_path} and {target_path}")

def save_actor_critic_model(self, base_dir="saved models", sub_dir="ActorCritic", actor_file_name="actor", critic_file_name="critic"):
    # Get the repository root path
    repo_root = get_repository_root_path()
    if repo_root is None:
        print("Repository root not found. Using current directory as the base path.")
        repo_root = "."

    # Construct the full paths
    actor_full_path = os.path.join(repo_root, base_dir, sub_dir, actor_file_name)
    critic_full_path = os.path.join(repo_root, base_dir, sub_dir, critic_file_name)

    # Ensure the directories exist
    os.makedirs(os.path.dirname(actor_full_path), exist_ok=True)
    os.makedirs(os.path.dirname(critic_full_path), exist_ok=True)

    # Define the full file paths for the actor and critic models
    actor_path = f"{actor_full_path}.pt"
    critic_path = f"{critic_full_path}.pt"

    # Save the models
    torch.save(self.actor.state_dict(), actor_path)
    torch.save(self.critic.state_dict(), critic_path)
    print(f"Models saved successfully: {actor_path} and {critic_path}")