# FrozenLake Q-Learning Agent

Ce projet utilise l'environnement FrozenLake-v1 de la bibliothèque OpenAI Gym pour implémenter un agent d'apprentissage par renforcement utilisant l'algorithme Q-learning. L'objectif est d'entraîner l'agent à naviguer dans un lac gelé en évitant les trous et en atteignant l'objectif avec succès.

## Fonctionnalités

- **Environnement Gym** : Utilisation de l'environnement FrozenLake-v1 avec ou sans glissade (`is_slippery`).
- **Q-Learning** : Implémentation de l'algorithme Q-learning pour entraîner l'agent à prendre des décisions optimales.
- **Visualisation** : Affichage de l'environnement et des résultats de l'agent au fur et à mesure de son apprentissage.
- **Hyperparamètres ajustables** : Possibilité de modifier les hyperparamètres tels que le taux d'apprentissage (`alpha`), le facteur de discount (`gamma`) et l'exploration (`epsilon`).

## Dépendances

- `gym` : Bibliothèque pour les environnements d'apprentissage par renforcement.
- `numpy` : Bibliothèque pour le calcul numérique.
- `matplotlib` : Bibliothèque pour la visualisation des résultats.

## Structure du Code

1. **Initialisation de l'environnement** :

   ```python
   env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
   env.reset()
   ```

2. **Initialisation de la Q-table** :

   ```python
   qtable = np.zeros((env.observation_space.n, env.action_space.n))
   ```

3. **Définition des hyperparamètres** :

   ```python
   episodes = 1000
   alpha = 0.5
   gamma = 0.9
   epsilon = 1.0
   epsilon_decay = 0.001
   ```

4. **Entraînement de l'agent** :

   - Pour chaque épisode, l'agent choisit une action basée sur la politique epsilon-greedy.
   - L'agent met à jour la Q-table selon la règle de mise à jour de Q-learning.
   - Epsilon est ajusté pour réduire progressivement l'exploration.

5. **Évaluation de l'agent** :

   - L'agent est évalué sur un certain nombre d'épisodes pour déterminer son taux de succès.

6. **Visualisation des résultats** :

   - Affichage de la Q-table avant et après l'entraînement.
   - Visualisation des résultats sous forme de graphique des succès et des échecs.

## Utilisation

1. Installez les dépendances nécessaires :

   ```bash
   pip install gym gym[toy_text] numpy matplotlib
   ```

2. Exécutez le script pour entraîner et évaluer l'agent :

   ```python
   python frozen_lake_q_learning.py
   ```

3. Visualisez les résultats et l'agent en action.

## Code

```python
import gym
import random
import numpy as np
import matplotlib.pyplot as plt

# Initialisation de l'environnement
env = gym.make("FrozenLake-v1", render_mode= "rgb_array", is_slippery=False)
env.reset()

# Initialisation de la Q-table
qtable = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparamètres
episodes = 1000
alpha = 0.5
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.001

# Entraînement
outcomes = []
for _ in range(episodes):
    state, _ = env.reset()
    done = False
    outcomes.append("Failure")
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state])
        new_state, reward, done, _, _ = env.step(action)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        state = new_state
        if reward:
            outcomes[-1] = "Success"
    epsilon = max(epsilon - epsilon_decay, 0)

# Évaluation
nb_success = 0
for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(qtable[state])
        new_state, reward, done, _, _ = env.step(action)
        state = new_state
        nb_success += reward

print(f"Success rate = {nb_success/100*100}%")

# Visualisation des résultats
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()
```
